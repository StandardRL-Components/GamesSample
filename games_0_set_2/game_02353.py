
# Generated: 2025-08-28T04:31:08.232426
# Source Brief: brief_02353.md
# Brief Index: 2353

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    game_description = (
        "Survive waves of zombies in an isometric arena. Manage your health and ammo to clear all 5 waves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 120 # 2 minutes max
        self.MAX_WAVES = 5

        # World/Isometric Dimensions
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 30, 30
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = self.TILE_WIDTH / 2
        self.ISO_ORIGIN_X = self.WIDTH / 2
        self.ISO_ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_PLAYER_SHADOW = (20, 20, 25)
        self.COLOR_ZOMBIE = (80, 150, 50)
        self.COLOR_ZOMBIE_SHADOW = (20, 20, 25)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_BLOOD = (180, 20, 20)
        self.COLOR_MUZZLE_FLASH = (255, 220, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_BAR_FG = (220, 50, 50)
        self.COLOR_AMMO_BAR_BG = (80, 80, 20)
        self.COLOR_AMMO_BAR_FG = (220, 220, 50)

        # Game Parameters
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 30
        self.PLAYER_SPEED = 0.6
        self.PLAYER_RADIUS = 0.5
        self.SHOOT_COOLDOWN = 5 # frames
        self.RELOAD_TIME = 60 # frames
        self.PROJECTILE_SPEED = 2.0
        self.PROJECTILE_RADIUS = 0.2
        self.ZOMBIE_BASE_SPEED = 0.2
        self.ZOMBIE_HEALTH = 1
        self.ZOMBIE_RADIUS = 0.6
        self.ZOMBIE_DAMAGE = 10
        self.WAVE_TRANSITION_TIME = 90 # frames

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.SysFont("monospace", 16, bold=True)
        self.FONT_WAVE = pygame.font.SysFont("impact", 48)
        self.FONT_GAMEOVER = pygame.font.SysFont("impact", 64)

        # --- State Variables ---
        self.player = {}
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.wave_transition_timer = 0
        self.game_over = False
        self.game_won = False
        self.screen_shake = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave = 1
        self.wave_transition_timer = self.WAVE_TRANSITION_TIME
        self.game_over = False
        self.game_won = False
        self.screen_shake = 0

        self.player = {
            "pos": np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float64),
            "health": self.PLAYER_MAX_HEALTH,
            "ammo": self.PLAYER_MAX_AMMO,
            "shoot_cooldown": 0,
            "reload_timer": 0,
            "is_reloading": False,
            "last_move_dir": np.array([0, -1.0]), # Face up
        }

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01 # Cost of living

        # --- Handle Wave Progression ---
        if not self.zombies and not self.game_won:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                self.wave += 1
                if self.wave > self.MAX_WAVES:
                    self.game_won = True
                    self.game_over = True
                    reward += 100.0
                else:
                    reward += 10.0
                    self._spawn_wave()
                    self.wave_transition_timer = self.WAVE_TRANSITION_TIME
        else:
            # --- Handle Player Input ---
            self._handle_input(action)

            # --- Update Game State ---
            self._update_player()
            self._update_projectiles()
            self._update_zombies()
            
            # --- Handle Collisions ---
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

        self._update_particles()

        # --- Update Timers & Cooldowns ---
        if self.player["shoot_cooldown"] > 0: self.player["shoot_cooldown"] -= 1
        if self.player["is_reloading"]:
            self.player["reload_timer"] -= 1
            if self.player["reload_timer"] <= 0:
                self.player["is_reloading"] = False
                self.player["ammo"] = self.PLAYER_MAX_AMMO
                # Sound: Reload complete

        if self.screen_shake > 0: self.screen_shake -= 1
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_vec = np.array([0, 0], dtype=np.float64)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        if np.linalg.norm(move_vec) > 0:
            self.player["last_move_dir"] = move_vec / np.linalg.norm(move_vec)
            self.player["pos"] += self.player["last_move_dir"] * self.PLAYER_SPEED
        
        # Shooting
        if space_held and self.player["shoot_cooldown"] <= 0 and self.player["ammo"] > 0 and not self.player["is_reloading"]:
            self._shoot()

        # Reloading
        if shift_held and not self.player["is_reloading"] and self.player["ammo"] < self.PLAYER_MAX_AMMO:
            self.player["is_reloading"] = True
            self.player["reload_timer"] = self.RELOAD_TIME
            # Sound: Reload start

    def _update_player(self):
        # Boundary checks
        self.player["pos"][0] = np.clip(self.player["pos"][0], 0, self.WORLD_WIDTH)
        self.player["pos"][1] = np.clip(self.player["pos"][1], 0, self.WORLD_HEIGHT)

    def _shoot(self):
        self.player["ammo"] -= 1
        self.player["shoot_cooldown"] = self.SHOOT_COOLDOWN
        
        proj_pos = self.player["pos"] + self.player["last_move_dir"] * (self.PLAYER_RADIUS + 0.1)
        self.projectiles.append({"pos": proj_pos, "vel": self.player["last_move_dir"]})
        
        # Muzzle flash
        flash_pos = proj_pos + self.player["last_move_dir"] * 0.5
        self.particles.append({"pos": flash_pos, "type": "muzzle", "life": 2, "size": 8})
        # Sound: Gunshot

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"] += p["vel"] * self.PROJECTILE_SPEED
        
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p["pos"][0] < self.WORLD_WIDTH and 0 < p["pos"][1] < self.WORLD_HEIGHT]

    def _update_zombies(self):
        zombie_speed = self.ZOMBIE_BASE_SPEED + (self.wave - 1) * 0.02
        for z in self.zombies:
            direction = self.player["pos"] - z["pos"]
            dist = np.linalg.norm(direction)
            if dist > 0:
                z["pos"] += (direction / dist) * zombie_speed

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Zombie
        zombies_hit_indices = set()
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            for j, z in enumerate(self.zombies):
                if j in zombies_hit_indices: continue
                if np.linalg.norm(p["pos"] - z["pos"]) < self.ZOMBIE_RADIUS + self.PROJECTILE_RADIUS:
                    z["health"] -= 1
                    projectiles_to_remove.append(i)
                    reward += 0.1 # Hit reward
                    # Blood Splatter
                    for _ in range(5):
                        self.particles.append({"pos": z["pos"].copy(), "type": "blood", "life": 20, "size": self.np_random.integers(2, 5), "vel": self.np_random.random(2) * 2 - 1})
                    
                    if z["health"] <= 0:
                        zombies_hit_indices.add(j)
                        reward += 1.0 # Kill reward
                        # Sound: Zombie death
                        # More blood on kill
                        for _ in range(15):
                            self.particles.append({"pos": z["pos"].copy(), "type": "blood", "life": 30, "size": self.np_random.integers(3, 7), "vel": self.np_random.random(2) * 3 - 2})
                    else:
                        # Sound: Zombie hit
                        pass
                    break # Projectile can only hit one zombie

        if zombies_hit_indices:
            self.zombies = [z for j, z in enumerate(self.zombies) if j not in zombies_hit_indices]
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in set(projectiles_to_remove)]

        # Player-Zombie
        for z in self.zombies:
            if np.linalg.norm(self.player["pos"] - z["pos"]) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                self.player["health"] -= self.ZOMBIE_DAMAGE
                self.screen_shake = 10
                # Sound: Player hurt
                # Pushback effect (optional)
                push_dir = self.player["pos"] - z["pos"]
                dist = np.linalg.norm(push_dir)
                if dist > 0:
                    self.player["pos"] += (push_dir / dist) * 1.5

        self.player["health"] = max(0, self.player["health"])
        return reward

    def _update_particles(self):
        for p in self.particles:
            p["life"] -= 1
            if p["type"] == "blood":
                p["pos"] += p["vel"] * 0.1
                p["vel"] *= 0.9
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _spawn_wave(self):
        num_zombies = 5 + (self.wave - 1) * 2
        for _ in range(num_zombies):
            # Spawn on edges
            edge = self.np_random.integers(0, 4)
            pos = np.zeros(2, dtype=np.float64)
            if edge == 0: # Top
                pos[0], pos[1] = self.np_random.uniform(0, self.WORLD_WIDTH), -1
            elif edge == 1: # Bottom
                pos[0], pos[1] = self.np_random.uniform(0, self.WORLD_WIDTH), self.WORLD_HEIGHT + 1
            elif edge == 2: # Left
                pos[0], pos[1] = -1, self.np_random.uniform(0, self.WORLD_HEIGHT)
            else: # Right
                pos[0], pos[1] = self.WORLD_WIDTH + 1, self.np_random.uniform(0, self.WORLD_HEIGHT)
            
            self.zombies.append({"pos": pos, "health": self.ZOMBIE_HEALTH})

    def _check_termination(self):
        return self.player["health"] <= 0 or self.steps >= self.MAX_STEPS or self.game_won

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player["health"],
            "ammo": self.player["ammo"],
            "zombies_left": len(self.zombies)
        }

    def _world_to_iso(self, pos):
        iso_x = self.ISO_ORIGIN_X + (pos[0] - pos[1]) * self.TILE_WIDTH / 2
        iso_y = self.ISO_ORIGIN_Y + (pos[0] + pos[1]) * self.TILE_HEIGHT / 2
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-5, 6)
            offset_y = self.np_random.integers(-5, 6)
        
        # --- Drawing ---
        self.screen.fill(self.COLOR_BG)
        
        # 1. Render Grid
        for i in range(self.WORLD_WIDTH + 1):
            start = self._world_to_iso(np.array([i, 0]))
            end = self._world_to_iso(np.array([i, self.WORLD_HEIGHT]))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.WORLD_HEIGHT + 1):
            start = self._world_to_iso(np.array([0, i]))
            end = self._world_to_iso(np.array([self.WORLD_WIDTH, i]))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # 2. Collect and sort all dynamic entities by Y-position for correct draw order
        render_list = []
        for z in self.zombies:
            render_list.append({"pos": z["pos"], "type": "zombie"})
        render_list.append({"pos": self.player["pos"], "type": "player"})
        
        render_list.sort(key=lambda item: item["pos"][1])

        # 3. Render sorted entities (shadows first, then bodies)
        for item in render_list:
            iso_pos = self._world_to_iso(item["pos"])
            if item["type"] == "player":
                shadow_radius = int(self.PLAYER_RADIUS * self.TILE_WIDTH / 2)
                pygame.gfxdraw.filled_ellipse(self.screen, iso_pos[0], iso_pos[1] + 8, shadow_radius, int(shadow_radius/2), self.COLOR_PLAYER_SHADOW)
            elif item["type"] == "zombie":
                shadow_radius = int(self.ZOMBIE_RADIUS * self.TILE_WIDTH / 2)
                pygame.gfxdraw.filled_ellipse(self.screen, iso_pos[0], iso_pos[1] + 8, shadow_radius, int(shadow_radius/2), self.COLOR_ZOMBIE_SHADOW)
        
        for item in render_list:
            iso_pos = self._world_to_iso(item["pos"])
            if item["type"] == "player":
                radius = int(self.PLAYER_RADIUS * self.TILE_WIDTH / 2)
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], radius, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], radius, self.COLOR_PLAYER)
            elif item["type"] == "zombie":
                radius = int(self.ZOMBIE_RADIUS * self.TILE_WIDTH / 2)
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], radius, self.COLOR_ZOMBIE)
                pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], radius, self.COLOR_ZOMBIE)

        # 4. Render particles and projectiles (on top)
        for p in self.particles:
            iso_pos = self._world_to_iso(p["pos"])
            if p["type"] == "blood":
                color = self.COLOR_BLOOD + (int(255 * (p["life"] / 30.0)),)
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], p["size"], color)
            elif p["type"] == "muzzle":
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], p["size"], self.COLOR_MUZZLE_FLASH)
        
        for p in self.projectiles:
            iso_pos = self._world_to_iso(p["pos"])
            radius = int(self.PROJECTILE_RADIUS * self.TILE_WIDTH / 2)
            pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], max(1, radius), self.COLOR_PROJECTILE)

        # 5. Render UI
        self._render_ui()

        # Final screen manipulation
        final_surface = self.screen
        if offset_x != 0 or offset_y != 0:
            final_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
            final_surface.fill(self.COLOR_BG)
            final_surface.blit(self.screen, (offset_x, offset_y))

        arr = pygame.surfarray.array3d(final_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player["health"] / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(200 * health_ratio), 20))
        health_text = self.FONT_UI.render(f"HP: {self.player['health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo
        ammo_text = self.FONT_UI.render(f"AMMO: {self.player['ammo']}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))
        if self.player["is_reloading"]:
            reload_ratio = self.player["reload_timer"] / self.RELOAD_TIME
            pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_BG, (10, 55, 100, 10))
            pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_FG, (10, 55, int(100 * (1-reload_ratio)), 10))
        
        # Wave Text
        wave_str = f"WAVE {self.wave}"
        if not self.zombies and not self.game_won:
            if self.wave < self.MAX_WAVES:
                wave_str = f"WAVE {self.wave + 1} INCOMING"
            else:
                wave_str = "FINAL WAVE CLEARED"
        
        wave_surf = self.FONT_WAVE.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.WIDTH / 2 - wave_surf.get_width() / 2, 10))

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU SURVIVED" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_surf = self.FONT_GAMEOVER.render(msg, True, color)
            self.screen.blit(end_surf, (self.WIDTH / 2 - end_surf.get_width() / 2, self.HEIGHT / 2 - end_surf.get_height() / 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()