﻿using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization.Json;
using System.IO;

namespace FishData
{
    class Program
    {
        public static async Task<bool> UpdateBatchFishLocation(string InstallationName, PonicsService.PonicsServicePonicsFishLocation[] fishLocs)
        {
            PonicsService.PonicsServiceClient ponicsSvc = new PonicsService.PonicsServiceClient();

            try
            {
                await ponicsSvc.LogBatchFishLocationAsync(InstallationName, fishLocs);
            }
            catch (Exception ex)
            {
                Debug.WriteLine("fail in UpdateBatchFishLocation : " + ex.Message);
                return false;
            }
            
            return true;
        }

        static void Main(string[] args)
        {
            if(args.Length < 1)
            {
                Console.WriteLine("Usage: FishData.exe \"[{\"FishId\":\"RedFish\",\"FishLocationDateTime\":{\"DateTime\":\"Date(-62135596800000)\",\"OffsetMinutes\":0},\"XPos\":1.234,\"YPos\":2.342,\"ZPos\":2.4445},{\"FishId\":\"BlueFish\",\"FishLocationDateTime\":{\"DateTime\":\"\\/Date(-62135596800000)\\/\",\"OffsetMinutes\":0},\"XPos\":1.234,\"YPos\":2.342,\"ZPos\":2.4445}]\"");
                return;
            }
            // parse out the locations
            string locationString = args[0];
            locationString = locationString.Replace('\'', '"');
            //Console.WriteLine("Using [" + locationString + "] as my data\n");
            PonicsService.PonicsServicePonicsFishLocation[] fishLocations;
            DataContractJsonSerializer jsonSerializer = new DataContractJsonSerializer(typeof(PonicsService.PonicsServicePonicsFishLocation[]));
            MemoryStream stream = new MemoryStream(Encoding.UTF8.GetBytes(locationString));
            object objResponse = jsonSerializer.ReadObject(stream);
            fishLocations = objResponse as PonicsService.PonicsServicePonicsFishLocation[];
            foreach(var loc in fishLocations)
            {
                loc.FishLocationDateTime = DateTime.Now;                
                PrintLocation(loc);
            }

            Console.WriteLine("Submitting locations to Azure.\n");
            var task = UpdateBatchFishLocation("99", fishLocations);

            Task.WaitAll(task);
            Console.WriteLine("Done!\n");
        }

        private static void PrintLocation(PonicsService.PonicsServicePonicsFishLocation loc)
        {
            Console.WriteLine("~~Location~~\n");
            Console.WriteLine("("+loc.XPos.ToString()+","+loc.YPos.ToString()+","+loc.ZPos.ToString()+")\n");
            Console.WriteLine(loc.FishLocationDateTime.ToString());
        }
    }
}

/* Schema of PonicsService.PonicsServicePonicsFishLocation
FishId - string
FishLocationDateTime - DateTimeOffset
XPos - double
YPos - double
ZPos - double
*/