import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:30:13.934975
# Source Brief: brief_01101.md
# Brief Index: 1101
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your cities from incoming ballistic missiles by firing anti-missiles from your launch sites."
    )
    user_guide = (
        "Controls: ←→ to move the reticle, ↑↓ to select a launch site. Press space to fire an anti-missile."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 360
    MAX_STEPS = 3000
    WIN_ROUND = 20

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GROUND = (40, 60, 40)
    COLOR_CITY_HEALTHY = (0, 200, 100)
    COLOR_CITY_DAMAGED = (200, 180, 0)
    COLOR_CITY_CRITICAL = (200, 50, 50)
    COLOR_ENEMY_MISSILE = (255, 50, 50)
    COLOR_PLAYER_MISSILE = (100, 150, 255)
    COLOR_EXPLOSION = (255, 150, 0)
    COLOR_RETICLE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_AMMO = (100, 150, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.round_num = 0
        self.cities = []
        self.launch_sites = []
        self.enemy_missiles = []
        self.player_missiles = []
        self.explosions = []
        self.reticle_x = self.SCREEN_WIDTH // 2
        self.selected_site_idx = 0
        self.prev_movement_action = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.round_num = 0

        # Initialize cities
        city_hps = [10, 15, 10]
        self.cities = [
            {
                "pos": (100 + i * 220, self.GROUND_Y),
                "max_hp": hp,
                "hp": hp,
                "width": 60,
            }
            for i, hp in enumerate(city_hps)
        ]

        # Initialize launch sites
        self.launch_sites = [
            {"pos": (50, self.GROUND_Y), "ammo": 0},
            {"pos": (self.SCREEN_WIDTH // 2, self.GROUND_Y), "ammo": 0},
            {"pos": (self.SCREEN_WIDTH - 50, self.GROUND_Y), "ammo": 0},
        ]
        self.selected_site_idx = 1

        self.enemy_missiles = []
        self.player_missiles = []
        self.explosions = []

        self.reticle_x = self.SCREEN_WIDTH // 2
        self.prev_movement_action = 0
        self.prev_space_held = False

        self._start_new_round()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_player_missiles()
            self._update_enemy_missiles()
            reward += self._update_explosions()

            # Check for round end
            if not self.enemy_missiles and not self.game_over:
                reward += 5  # Survived the round
                self._start_new_round()

            terminated = self._check_termination()
            if terminated:
                reward += 100 if self.win else -100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _start_new_round(self):
        self.round_num += 1
        if self.round_num > self.WIN_ROUND:
            self.win = True
            self.game_over = True
            return

        # Replenish ammo
        for site in self.launch_sites:
            site["ammo"] = 5

        # Spawn enemy missiles
        num_missiles = 3 + (self.round_num - 1) // 5
        
        # Valid targets are living cities
        valid_targets = [c for c in self.cities if c["hp"] > 0]
        if not valid_targets: # No targets left
            return

        for _ in range(num_missiles):
            start_x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            target_city = self.np_random.choice(valid_targets)
            target_x = target_city["pos"][0] + self.np_random.uniform(-target_city["width"]//4, target_city["width"]//4)
            
            # Add some randomness to missile speed
            travel_time = self.np_random.uniform(200, 300) - self.round_num * 5
            
            self.enemy_missiles.append({
                "start_pos": np.array([start_x, 0.0]),
                "end_pos": np.array([target_x, float(self.GROUND_Y)]),
                "pos": np.array([start_x, 0.0]),
                "arc_height": self.np_random.uniform(50, 150),
                "timer": 0,
                "travel_time": max(60, travel_time), # Minimum travel time
                "trail": [],
            })

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Reticle movement (continuous)
        if movement == 3:  # Left
            self.reticle_x = max(0, self.reticle_x - 8)
        elif movement == 4:  # Right
            self.reticle_x = min(self.SCREEN_WIDTH, self.reticle_x + 8)

        # Launch site selection (on press)
        if movement in [1, 2] and movement != self.prev_movement_action:
            if movement == 1:  # Up
                self.selected_site_idx = (self.selected_site_idx + 1) % len(self.launch_sites)
            elif movement == 2:  # Down
                self.selected_site_idx = (self.selected_site_idx - 1 + len(self.launch_sites)) % len(self.launch_sites)
        
        # Launch anti-missile (on press)
        if space_held and not self.prev_space_held:
            site = self.launch_sites[self.selected_site_idx]
            if site["ammo"] > 0:
                site["ammo"] -= 1
                start_pos = np.array(site["pos"], dtype=float)
                # Target just above ground
                target_pos = np.array([float(self.reticle_x), float(self.GROUND_Y - 250 + self.np_random.uniform(-30, 30))], dtype=float)
                
                # Anti-missiles are fast
                travel_time = 50 
                
                self.player_missiles.append({
                    "start_pos": start_pos,
                    "end_pos": target_pos,
                    "pos": start_pos.copy(),
                    "timer": 0,
                    "travel_time": travel_time,
                    "trail": []
                })
                # sfx: player missile launch

        self.prev_movement_action = movement
        self.prev_space_held = space_held

    def _update_player_missiles(self):
        for m in self.player_missiles[:]:
            m["timer"] += 1
            progress = min(1.0, m["timer"] / m["travel_time"])
            
            m["trail"].append(m["pos"].copy())
            if len(m["trail"]) > 10:
                m["trail"].pop(0)

            m["pos"] = m["start_pos"] + (m["end_pos"] - m["start_pos"]) * progress

            if progress >= 1.0:
                self.player_missiles.remove(m)
                self.explosions.append({
                    "pos": m["pos"],
                    "radius": 0,
                    "max_radius": 40,
                    "life": 0,
                    "max_life": 30, # frames
                })
                # sfx: player explosion

    def _update_enemy_missiles(self):
        for m in self.enemy_missiles[:]:
            m["timer"] += 1
            progress = min(1.0, m["timer"] / m["travel_time"])

            m["trail"].append(m["pos"].copy())
            if len(m["trail"]) > 15:
                m["trail"].pop(0)

            # Linear interpolation for position
            x = m["start_pos"][0] + (m["end_pos"][0] - m["start_pos"][0]) * progress
            y_linear = m["start_pos"][1] + (m["end_pos"][1] - m["start_pos"][1]) * progress
            # Add sinusoidal arc
            y_arc = m["arc_height"] * math.sin(progress * math.pi)
            m["pos"] = np.array([x, y_linear - y_arc])

            if progress >= 1.0:
                self.enemy_missiles.remove(m)
                # Check for city collision
                for city in self.cities:
                    if city["hp"] > 0 and abs(m["end_pos"][0] - city["pos"][0]) < city["width"] / 2:
                        city["hp"] -= 5 # Damage
                        # sfx: city damage
                        # Create small ground explosion
                        self.explosions.append({
                            "pos": m["end_pos"], "radius": 0, "max_radius": 20, "life": 0, "max_life": 15
                        })
                        break

    def _update_explosions(self):
        reward = 0
        for e in self.explosions[:]:
            e["life"] += 1
            progress = e["life"] / e["max_life"]
            e["radius"] = progress * e["max_radius"]

            if e["life"] >= e["max_life"]:
                self.explosions.remove(e)
                continue
            
            # Check for collision with enemy missiles
            for m in self.enemy_missiles[:]:
                dist = np.linalg.norm(m["pos"] - e["pos"])
                if dist < e["radius"]:
                    self.enemy_missiles.remove(m)
                    reward += 1
                    # sfx: enemy missile destroyed
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Lose condition: all cities destroyed
        if all(c["hp"] <= 0 for c in self.cities):
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # --- Game Elements ---
        self._render_cities()
        self._render_launch_sites()
        self._render_missiles(self.enemy_missiles, self.COLOR_ENEMY_MISSILE)
        self._render_missiles(self.player_missiles, self.COLOR_PLAYER_MISSILE)
        self._render_explosions()
        self._render_reticle()

        # --- UI ---
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_cities(self):
        for city in self.cities:
            if city["hp"] <= 0:
                continue
            
            hp_ratio = city["hp"] / city["max_hp"]
            height = max(1, 20 * hp_ratio)
            
            if hp_ratio > 0.6: color = self.COLOR_CITY_HEALTHY
            elif hp_ratio > 0.3: color = self.COLOR_CITY_DAMAGED
            else: color = self.COLOR_CITY_CRITICAL
            
            x, y = city["pos"]
            w = city["width"]
            pygame.draw.rect(self.screen, color, (x - w / 2, y - height, w, height))
            
            hp_text = self.font_small.render(str(city['hp']), True, self.COLOR_UI_TEXT)
            self.screen.blit(hp_text, (x - hp_text.get_width() / 2, y - height - 20))


    def _render_launch_sites(self):
        for i, site in enumerate(self.launch_sites):
            x, y = site["pos"]
            is_selected = (i == self.selected_site_idx)
            color = self.COLOR_PLAYER_MISSILE if is_selected else self.COLOR_GROUND
            
            # Draw base
            pygame.gfxdraw.filled_polygon(self.screen, [(x-10, y), (x+10, y), (x, y-10)], color)
            
            # Draw ammo
            ammo_text = self.font_small.render(f"A:{site['ammo']}", True, self.COLOR_UI_AMMO)
            self.screen.blit(ammo_text, (x - ammo_text.get_width() / 2, y + 5))

    def _render_missiles(self, missiles, color):
        for m in missiles:
            # Draw trail
            if len(m["trail"]) > 1:
                for i in range(len(m["trail"]) - 1):
                    alpha = int(255 * (i / len(m["trail"])))
                    start = tuple(map(int, m["trail"][i]))
                    end = tuple(map(int, m["trail"][i+1]))
                    pygame.gfxdraw.line(self.screen, start[0], start[1], end[0], end[1], (*color, alpha))
            
            # Draw head
            pygame.gfxdraw.filled_circle(self.screen, int(m["pos"][0]), int(m["pos"][1]), 3, color)

    def _render_explosions(self):
        for e in self.explosions:
            progress = e["life"] / e["max_life"]
            alpha = int(255 * (1 - progress**2))
            color = (*self.COLOR_EXPLOSION, alpha)
            radius = int(e["radius"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(e["pos"][0]), int(e["pos"][1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(e["pos"][0]), int(e["pos"][1]), radius, color)

    def _render_reticle(self):
        x = int(self.reticle_x)
        
        # Determine y position for the reticle. This logic seems complex and might be simplified.
        # It tries to place the reticle at the target y of the last fired missile if it's still in flight.
        # Otherwise, it defaults to a fixed height above the selected launch site.
        # A simpler approach might be to always use a fixed y or the mouse y-position in interactive mode.
        # For the agent, the y-position of the explosion is determined by the missile's target_pos, not the reticle's y.
        # Let's keep the original logic for now.
        default_y = self.launch_sites[self.selected_site_idx]['pos'][1] - 250
        y = int(default_y)
        if self.player_missiles:
            last_missile = self.player_missiles[-1]
            if last_missile['timer'] < last_missile['travel_time']:
                y = int(last_missile['end_pos'][1])

        # Simple crosshair
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 10, y), (x + 10, y), 1)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - 10), (x, y + 10), 1)


    def _render_ui(self):
        # Round Number
        round_text = self.font_large.render(f"ROUND {self.round_num}/{self.WIN_ROUND}", True, self.COLOR_UI_TEXT)
        self.screen.blit(round_text, (10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_CITY_HEALTHY if self.win else self.COLOR_ENEMY_MISSILE
            end_text = pygame.font.SysFont("monospace", 72, bold=True).render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "round": self.round_num,
            "ammo": sum(s["ammo"] for s in self.launch_sites),
            "cities_hp": [c["hp"] for c in self.cities],
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Missile Command Gym Env")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward}, Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()