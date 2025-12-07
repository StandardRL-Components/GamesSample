import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:10:38.176230
# Source Brief: brief_02149.md
# Brief Index: 2149
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for Rocket Ranch, a retro-futuristic resource management game.
    The player manages a fleet of rockets, teleporting them between planets to gather
    resources while avoiding asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a fleet of rockets to gather resources from different planets "
        "while dodging asteroids in this retro-futuristic management game."
    )
    user_guide = (
        "Controls: Use keys 1-4 to select a rocket. Press Shift to cycle through "
        "target planets and Space to teleport the selected rocket."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (200, 200, 220)
    COLOR_RANCH = (50, 60, 80)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_UI_BG = (40, 45, 70)
    COLOR_FUEL = (100, 255, 120)
    COLOR_FUEL_BG = (255, 100, 100)
    COLOR_RESOURCE = (100, 180, 255)
    COLOR_ROCKET = (255, 220, 50)
    COLOR_ROCKET_GLOW = (255, 240, 150)
    COLOR_ROCKET_DMG = (200, 50, 50)
    COLOR_ASTEROID = (255, 100, 80)
    COLOR_TARGET = (255, 255, 255)
    
    PLANET_DEFS = [
        {"name": "Home Base", "color": (80, 120, 255), "radius": 30, "pos": (120, 200), "gen_rate": 0.2},
        {"name": "Mars", "color": (220, 100, 80), "radius": 25, "pos": (500, 100), "gen_rate": 0.3},
        {"name": "Cryo", "color": (150, 220, 250), "radius": 28, "pos": (540, 300), "gen_rate": 0.4},
        {"name": "Gas Giant", "color": (200, 180, 140), "radius": 40, "pos": (450, 200), "gen_rate": 0.5},
    ]
    
    ROCKET_UNLOCK_THRESHOLDS = [0, 100, 250, 500] # Unlocks at 0, then 100, etc. resources
    PLANET_UNLOCK_THRESHOLDS = [0, 100, 250, 500]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_planet = pygame.font.Font(None, 18)
        self.font_msg = pygame.font.Font(None, 32)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = 0.0
        self.resources = 0
        self.asteroids = []
        self.particles = []
        self.rockets = []
        self.planets = []
        self.stars = []
        self.selected_rocket_idx = 0
        self.target_planet_idx = 0
        self.prev_shift_state = 0
        self.prev_space_state = 0
        self.last_resource_milestone = 0
        self.asteroid_spawn_chance = 0.05 # Initial: 1 every 20 steps
        self.message = ""
        self.message_timer = 0
        
        # Initialize state by calling reset
        # self.reset() # reset is called by the wrapper
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = 2000.0
        self.resources = 0
        self.last_resource_milestone = 0
        self.selected_rocket_idx = 0
        self.target_planet_idx = 0
        self.prev_shift_state = 0
        self.prev_space_state = 0
        self.asteroid_spawn_chance = 0.05
        self.message = ""
        self.message_timer = 0
        
        self.asteroids = []
        self.particles = []
        
        # Initialize planets
        self.planets = []
        for i, p_def in enumerate(self.PLANET_DEFS):
            self.planets.append({
                **p_def,
                "resources": 10 if i == 0 else 0,
                "unlocked": self.resources >= self.PLANET_UNLOCK_THRESHOLDS[i]
            })

        # Initialize rockets
        self.rockets = []
        for i in range(len(self.ROCKET_UNLOCK_THRESHOLDS)):
            dock_pos = (50, 80 + i * 80)
            self.rockets.append({
                "pos": np.array(dock_pos, dtype=float),
                "dock_pos": np.array(dock_pos, dtype=float),
                "unlocked": self.resources >= self.ROCKET_UNLOCK_THRESHOLDS[i],
                "damaged": False,
                "teleport_anim": None,
            })
        
        # Generate a static starfield
        self.stars = []
        for _ in range(150):
            self.stars.append(
                ((self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                 self.np_random.integers(1, 3))
            )
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_binary, shift_binary = action
        space_held = space_binary == 1
        shift_held = shift_binary == 1

        # --- Handle Player Actions ---
        # 1. Select Rocket (Movement keys)
        if 1 <= movement < len(self.rockets) + 1:
            if self.rockets[movement - 1]["unlocked"]:
                self.selected_rocket_idx = movement - 1

        # 2. Cycle Target Planet (Shift key press)
        if shift_held and not self.prev_shift_state:
            # SFX: UI_Bleep
            unlocked_indices = [i for i, p in enumerate(self.planets) if p["unlocked"]]
            if unlocked_indices:
                current_list_idx = unlocked_indices.index(self.target_planet_idx) if self.target_planet_idx in unlocked_indices else -1
                next_list_idx = (current_list_idx + 1) % len(unlocked_indices)
                self.target_planet_idx = unlocked_indices[next_list_idx]

        # 3. Teleport (Space key press)
        if space_held and not self.prev_space_state:
            teleport_reward = self._handle_teleport()
            reward += teleport_reward

        self.prev_space_state = space_held
        self.prev_shift_state = shift_held

        # --- Update Game World ---
        self._update_planets()
        asteroid_reward = self._update_asteroids()
        reward += asteroid_reward
        self._update_particles()
        self._update_animations()
        
        progression_reward = self._update_progression()
        reward += progression_reward

        # --- Update Score & Milestones ---
        self.score += reward
        if self.resources // 100 > self.last_resource_milestone:
            self.last_resource_milestone = self.resources // 100
            reward += 10 # Milestone reward
            self._set_message(f"Resource Milestone: {self.resources // 100 * 100}!", 90)


        # --- Check Termination ---
        terminated = self.fuel <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and not self.game_over:
            self.game_over = True
            if self.fuel <= 0:
                reward -= 10 # Fuel exhaustion penalty
                self._set_message("OUT OF FUEL", 180)
            else:
                self._set_message("MAX STEPS REACHED", 180)

        if self.message_timer > 0:
            self.message_timer -= 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "resources": self.resources,
        }
        
    def _set_message(self, text, duration):
        self.message = text
        self.message_timer = duration

    # --- Game Logic Sub-routines ---
    
    def _handle_teleport(self):
        if self.selected_rocket_idx is None: return 0
        
        rocket = self.rockets[self.selected_rocket_idx]
        if not rocket["unlocked"] or rocket["damaged"] or rocket["teleport_anim"]:
            return 0
        
        planet = self.planets[self.target_planet_idx]
        if not planet["unlocked"]:
            return 0

        # SFX: Teleport_Initiate
        start_pos = rocket["pos"]
        end_pos = np.array(planet["pos"], dtype=float)
        distance = np.linalg.norm(end_pos - start_pos)
        fuel_cost = distance * 0.5

        if self.fuel < fuel_cost:
            self._set_message("INSUFFICIENT FUEL", 60)
            return 0
        
        self.fuel -= fuel_cost
        reward = -0.1 * fuel_cost
        
        # Collect resources
        collected = planet["resources"]
        if collected > 0:
            # SFX: Resource_Collect
            self.resources += collected
            planet["resources"] = 0
            reward += collected # +1 per resource
        
        rocket["teleport_anim"] = {
            "start_pos": start_pos,
            "end_pos": end_pos,
            "progress": 0.0,
            "duration": max(30, int(distance / 8)),
        }
        return reward

    def _update_planets(self):
        for planet in self.planets:
            if planet["unlocked"]:
                planet["resources"] = min(100, planet["resources"] + planet["gen_rate"])

    def _update_asteroids(self):
        reward = 0
        # Move existing asteroids and check for collisions
        for asteroid in self.asteroids[:]:
            asteroid["pos"] += asteroid["vel"]
            
            # Collision with rockets
            for rocket in self.rockets:
                if rocket["unlocked"] and not rocket["teleport_anim"]:
                    dist = np.linalg.norm(asteroid["pos"] - rocket["pos"])
                    if dist < asteroid["radius"] + 10: # 10 is rocket radius
                        if not rocket["damaged"]:
                            # SFX: Explosion_Hit
                            rocket["damaged"] = True
                            reward -= 5 # Penalty for damage
                            self._create_explosion(rocket["pos"])
                        self.asteroids.remove(asteroid)
                        break
            else: # If loop didn't break
                # Remove if off-screen
                if not (0 < asteroid["pos"][0] < self.SCREEN_WIDTH and 0 < asteroid["pos"][1] < self.SCREEN_HEIGHT):
                    self.asteroids.remove(asteroid)
        
        # Spawn new asteroids
        self.asteroid_spawn_chance = min(0.1, 0.05 + 0.001 * (self.steps / 100))
        if self.np_random.random() < self.asteroid_spawn_chance:
            self._spawn_asteroid()
            
        return reward

    def _spawn_asteroid(self):
        edge = self.np_random.choice(["top", "bottom", "right"])
        if edge == "top":
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -20.0])
            vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(2, 4)])
        elif edge == "bottom":
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20.0])
            vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-4, -2)])
        else: # right
            pos = np.array([self.SCREEN_WIDTH + 20.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
            vel = np.array([self.np_random.uniform(-4, -2), self.np_random.uniform(-1, 1)])
        
        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "radius": self.np_random.integers(8, 16),
            "angle": 0,
            "rot_speed": self.np_random.uniform(-0.1, 0.1)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
                
    def _update_animations(self):
        for rocket in self.rockets:
            if anim := rocket.get("teleport_anim"):
                anim["progress"] += 1.0 / anim["duration"]
                if anim["progress"] >= 1.0:
                    # SFX: Teleport_Arrive
                    rocket["pos"] = anim["end_pos"]
                    rocket["teleport_anim"] = None
                    # Teleporting to home base repairs the rocket
                    if np.array_equal(rocket["pos"], rocket["dock_pos"]):
                        rocket["damaged"] = False

    def _update_progression(self):
        reward = 0
        # Unlock Planets
        for i, p in enumerate(self.planets):
            if not p["unlocked"] and self.resources >= self.PLANET_UNLOCK_THRESHOLDS[i]:
                # SFX: Unlock_Success
                p["unlocked"] = True
                reward += 5
                self._set_message(f"PLANET {p['name']} UNLOCKED!", 120)

        # Unlock Rockets
        for i, r in enumerate(self.rockets):
            if not r["unlocked"] and self.resources >= self.ROCKET_UNLOCK_THRESHOLDS[i]:
                r["unlocked"] = True
                # No extra reward, tied to planet unlocks
        
        return reward
        
    def _create_explosion(self, pos, num_particles=50):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(20, 41)
            color = random.choice([(255, 100, 80), (255, 220, 50), (255, 255, 255)])
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "color": color})
            
    # --- Rendering Sub-routines ---

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_ranch_base()
        self._render_animations()
        self._render_planets()
        self._render_asteroids()
        self._render_rockets()
        self._render_particles()
        self._render_ui()
        self._render_messages()
        
    def _render_background(self):
        for pos, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (*pos, size, size))
            
    def _render_ranch_base(self):
        pygame.draw.rect(self.screen, self.COLOR_RANCH, (10, 50, 80, 300), border_radius=10)

    def _render_planets(self):
        for i, planet in enumerate(self.planets):
            if not planet["unlocked"]: continue
            
            pos = (int(planet["pos"][0]), int(planet["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], planet["radius"], planet["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], planet["radius"], tuple(c*0.8 for c in planet["color"]))
            
            # Resource text
            res_text = self.font_planet.render(f"{int(planet['resources'])}", True, self.COLOR_UI_TEXT)
            self.screen.blit(res_text, (pos[0] - res_text.get_width() // 2, pos[1] - planet["radius"] - 20))

            # Target reticle
            if i == self.target_planet_idx:
                r = planet["radius"] + 8
                pts = [(r,0), (r-5,0), (-r,0), (-r+5,0), (0,r), (0,r-5), (0,-r), (0,-r+5)]
                for k in range(4):
                    pygame.draw.line(self.screen, self.COLOR_TARGET, 
                                     (pos[0]+pts[2*k][0], pos[1]+pts[2*k][1]), 
                                     (pos[0]+pts[2*k+1][0], pos[1]+pts[2*k+1][1]), 1)

    def _render_rockets(self):
        for i, rocket in enumerate(self.rockets):
            if not rocket["unlocked"] or rocket["teleport_anim"]: continue

            pos = (int(rocket["pos"][0]), int(rocket["pos"][1]))
            
            # Selection glow
            if i == self.selected_rocket_idx:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_ROCKET_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_ROCKET_GLOW)
            
            # Rocket body
            points = [(pos[0], pos[1] - 12), (pos[0] - 7, pos[1] + 8), (pos[0] + 7, pos[1] + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ROCKET)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ROCKET)

            # Damage effect
            if rocket["damaged"]:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ROCKET_DMG)
                if self.steps % 10 < 5: # Smoke particles
                    p_pos = rocket["pos"] + np.array([self.np_random.uniform(-5,5), self.np_random.uniform(-5,5)])
                    p_vel = np.array([self.np_random.uniform(-0.5,0.5), self.np_random.uniform(-0.5,0.5)])
                    self.particles.append({"pos": p_pos, "vel": p_vel, "lifespan": 20, "color": (100,100,100)})

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["angle"] += asteroid["rot_speed"]
            points = []
            for i in range(6): # Hexagonal asteroids
                angle = asteroid["angle"] + (i * math.pi / 3)
                dist = asteroid["radius"] * (1 + self.np_random.uniform(-0.2, 0.2) if i%2==0 else 1)
                points.append((
                    int(asteroid["pos"][0] + dist * math.cos(angle)),
                    int(asteroid["pos"][1] + dist * math.sin(angle))
                ))
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_animations(self):
        for rocket in self.rockets:
            if anim := rocket.get("teleport_anim"):
                p = anim["progress"]
                curr_pos = anim["start_pos"] * (1 - p) + anim["end_pos"] * p
                
                # Trail
                pygame.draw.line(self.screen, self.COLOR_ROCKET_GLOW, 
                                 (int(anim["start_pos"][0]), int(anim["start_pos"][1])), 
                                 (int(curr_pos[0]), int(curr_pos[1])), 2)
                
                # Fading rocket head
                size = int(10 * (1 - abs(p - 0.5) * 2)) # Grow then shrink
                if size > 0:
                    color = (*self.COLOR_ROCKET, int(255 * (1-p)))
                    try: # gfxdraw can fail with transparent colors
                        pygame.gfxdraw.filled_circle(self.screen, int(curr_pos[0]), int(curr_pos[1]), size, color)
                        pygame.gfxdraw.aacircle(self.screen, int(curr_pos[0]), int(curr_pos[1]), size, color)
                    except TypeError:
                        pass


    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(3 * (p["lifespan"] / 40.0)))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), size, size))
            
    def _render_ui(self):
        # UI Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, 40))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (0, 40), (self.SCREEN_WIDTH, 40), 1)

        # Fuel
        fuel_text = self.font_ui.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (10, 12))
        fuel_ratio = max(0, self.fuel / 2000.0)
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BG, (70, 10, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_FUEL, (70, 10, int(150 * fuel_ratio), 20))

        # Resources
        res_text = self.font_ui.render("RESOURCES", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (240, 12))
        pygame.gfxdraw.filled_circle(self.screen, 350, 20, 10, self.COLOR_RESOURCE)
        pygame.gfxdraw.aacircle(self.screen, 350, 20, 10, self.COLOR_RESOURCE)
        res_val_text = self.font_ui.render(f"{self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_val_text, (370, 12))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 12))
        
    def _render_messages(self):
        if self.message_timer > 0:
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_UI_TEXT)
            pos = (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, 
                   self.SCREEN_HEIGHT - 50)
            self.screen.blit(msg_surf, pos)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Loop ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override Pygame display for direct rendering
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    pygame.display.set_caption("Rocket Ranch")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- CONTROLS ---")
    print("1, 2, 3, 4: Select Rocket")
    print("SHIFT: Cycle Target Planet")
    print("SPACE: Teleport Selected Rocket to Target")
    print("Q or ESC: Quit")
    print("----------------\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                # Update action state on key down
                if event.key == pygame.K_1: movement = 1
                elif event.key == pygame.K_2: movement = 2
                elif event.key == pygame.K_3: movement = 3
                elif event.key == pygame.K_4: movement = 4
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

            if event.type == pygame.KEYUP:
                # Reset action state on key up
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]: movement = 0
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0
        
        if not terminated:
            action = np.array([movement, space_held, shift_held])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # For single-press actions, reset after step
            movement = 0 
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()