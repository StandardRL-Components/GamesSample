
# Generated: 2025-08-28T03:56:35.878237
# Source Brief: brief_02163.md
# Brief Index: 2163

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrows to move cursor, Space to place selected tower, Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of zombies by placing attack towers. Earn money for kills to build more."
    )

    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_PATH = (55, 60, 70)
    COLOR_BASE = (0, 150, 136)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_ZOMBIE_SLOWED = (100, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255, 100)
    
    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 50, "range": 100, "cooldown": 30, "color": (255, 193, 7), "projectile_speed": 8, "damage": 10},
        1: {"name": "Frost", "cost": 75, "range": 70, "cooldown": 60, "color": (33, 150, 243), "projectile_speed": 6, "damage": 2, "slow": 0.5, "slow_duration": 90},
    }

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    MAX_WAVES = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        self.path_points = self._generate_path()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.money = 0
        self.wave_number = 0
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_manager = {}
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.money = 125
        self.wave_number = 0
        
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.SCREEN_WIDTH // (2 * self.GRID_SIZE), self.SCREEN_HEIGHT // (2 * self.GRID_SIZE)]
        self.selected_tower_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _generate_path(self):
        path = []
        p1 = (0, self.SCREEN_HEIGHT * 0.2)
        p2 = (self.SCREEN_WIDTH * 0.7, self.SCREEN_HEIGHT * 0.2)
        p3 = (self.SCREEN_WIDTH * 0.7, self.SCREEN_HEIGHT * 0.8)
        p4 = (self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.8)
        p5 = (self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.5)
        p6 = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.5)

        for i in range(int(p2[0] - p1[0])): path.append((p1[0] + i, p1[1]))
        for i in range(int(p3[1] - p2[1])): path.append((p2[0], p2[1] + i))
        for i in range(int(p2[0] - p4[0])): path.append((p3[0] - i, p3[1]))
        for i in range(int(p4[1] - p5[1])): path.append((p4[0], p4[1] - i))
        for i in range(int(p6[0] - p5[0])): path.append((p5[0] + i, p5[1]))
        
        return [(int(x), int(y)) for x, y in path]

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        num_zombies = 5 + (self.wave_number - 1) * 3
        base_health = 20 * (1.1 ** (self.wave_number - 1))
        base_speed = 1.0 * (1.05 ** (self.wave_number - 1))
        
        self.wave_manager = {
            "zombies_to_spawn": num_zombies,
            "spawn_timer": 0,
            "spawn_delay": 45, # 1.5 seconds between zombies
            "zombie_health": base_health,
            "zombie_speed": base_speed,
            "active": True
        }

    def step(self, action):
        reward = -0.001 # Small penalty for time passing
        self.steps += 1

        # --- 1. Handle Input ---
        self._handle_input(action)

        # --- 2. Update Game Logic ---
        self._update_wave_manager()
        self._update_towers()
        reward += self._update_zombies()
        reward += self._update_projectiles()
        self._update_particles()
        
        # --- 3. Check for Wave Completion ---
        if self.wave_manager["active"] is False and not self.zombies:
            if self.wave_number < self.MAX_WAVES:
                reward += 1.0
                self._start_next_wave()
        
        # --- 4. Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated:
            if self.base_health <= 0:
                reward -= 10 # Penalty for losing
            elif self.wave_number > self.MAX_WAVES:
                reward += 100 # Large reward for winning
                self.game_over = True # Ensure win message shows
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        # Movement for cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH // self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT // self.GRID_SIZE - 1)

        # Space press (rising edge) to place tower
        if space_action and not self.last_space_held:
            self._place_tower()
        self.last_space_held = space_action

        # Shift press (rising edge) to cycle tower type
        if shift_action and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_cycle.wav
        self.last_shift_held = shift_action

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.money >= spec["cost"]:
            cx, cy = self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE//2, self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE//2
            
            # Check if on path
            min_dist_to_path = min([math.hypot(cx - px, cy - py) for px, py in self.path_points])
            if min_dist_to_path < self.GRID_SIZE * 1.5:
                # sfx: error.wav
                return

            # Check if overlapping another tower
            for tower in self.towers:
                if math.hypot(cx - tower["pos"][0], cy - tower["pos"][1]) < self.GRID_SIZE:
                    # sfx: error.wav
                    return

            self.money -= spec["cost"]
            self.towers.append({
                "pos": (cx, cy),
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            })
            # sfx: place_tower.wav
            self._create_particles( (cx,cy), spec["color"], 20, 3, 15)

    def _update_wave_manager(self):
        if not self.wave_manager["active"]: return

        if self.wave_manager["zombies_to_spawn"] > 0:
            self.wave_manager["spawn_timer"] += 1
            if self.wave_manager["spawn_timer"] >= self.wave_manager["spawn_delay"]:
                self.wave_manager["spawn_timer"] = 0
                self.wave_manager["zombies_to_spawn"] -= 1
                self.zombies.append({
                    "path_index": 0,
                    "pos": self.path_points[0],
                    "health": self.wave_manager["zombie_health"],
                    "max_health": self.wave_manager["zombie_health"],
                    "speed": self.wave_manager["zombie_speed"],
                    "slow_timer": 0
                })
        elif self.wave_manager["zombies_to_spawn"] == 0:
            self.wave_manager["active"] = False

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Find a new target if needed
            if tower["target"] is None or tower["target"] not in self.zombies:
                tower["target"] = None
                possible_targets = [z for z in self.zombies if math.hypot(z["pos"][0] - tower["pos"][0], z["pos"][1] - tower["pos"][1]) <= spec["range"]]
                if possible_targets:
                    # Target zombie furthest along the path
                    tower["target"] = max(possible_targets, key=lambda z: z["path_index"])
            
            # Fire if target is in range
            if tower["target"] and tower["cooldown"] <= 0:
                dist = math.hypot(tower["target"]["pos"][0] - tower["pos"][0], tower["target"]["pos"][1] - tower["pos"][1])
                if dist <= spec["range"]:
                    self.projectiles.append({
                        "pos": list(tower["pos"]),
                        "type": tower["type"],
                        "target": tower["target"]
                    })
                    tower["cooldown"] = spec["cooldown"]
                    # sfx: tower_fire.wav

    def _update_zombies(self):
        reward_from_zombies = 0
        for z in self.zombies[:]:
            current_speed = z["speed"]
            if z["slow_timer"] > 0:
                z["slow_timer"] -= 1
                current_speed *= (1 - self.TOWER_SPECS[1]["slow"])
            
            for _ in range(int(current_speed)):
                if z["path_index"] < len(self.path_points) - 1:
                    z["path_index"] += 1
                else:
                    self.zombies.remove(z)
                    self.base_health -= 10
                    # sfx: base_damage.wav
                    self._create_particles(z["pos"], self.COLOR_BASE_DMG, 30, 4, 20)
                    break
            else: # continue if inner loop did not break
                z["pos"] = self.path_points[z["path_index"]]
            
            if z["health"] <= 0:
                self.zombies.remove(z)
                self.score += 10
                self.money += 15
                reward_from_zombies += 0.1
                # sfx: zombie_die.wav
                self._create_particles(z["pos"], self.COLOR_ZOMBIE, 15, 2, 10)
        
        return reward_from_zombies

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            spec = self.TOWER_SPECS[p["type"]]
            if p["target"] not in self.zombies:
                self.projectiles.remove(p)
                continue

            target_pos = p["target"]["pos"]
            px, py = p["pos"]
            tx, ty = target_pos
            angle = math.atan2(ty - py, tx - px)
            
            p["pos"][0] += math.cos(angle) * spec["projectile_speed"]
            p["pos"][1] += math.sin(angle) * spec["projectile_speed"]

            if math.hypot(p["pos"][0] - tx, p["pos"][1] - ty) < self.GRID_SIZE // 2:
                p["target"]["health"] -= spec["damage"]
                if "slow" in spec:
                    p["target"]["slow_timer"] = spec["slow_duration"]
                
                self.projectiles.remove(p)
                # sfx: projectile_hit.wav
                self._create_particles(p["pos"], spec["color"], 10, 1, 8)
        return 0

    def _update_particles(self):
        for particle in self.particles[:]:
            particle["pos"][0] += particle["vel"][0]
            particle["pos"][1] += particle["vel"][1]
            particle["vel"][1] += 0.1 # Gravity
            particle["life"] -= 1
            if particle["life"] <= 0:
                self.particles.remove(particle)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.MAX_WAVES and not self.zombies:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Draw path
        if len(self.path_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, self.GRID_SIZE*2)

        # Draw base
        base_pos = self.path_points[-1]
        base_rect = pygame.Rect(base_pos[0], base_pos[1] - self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE*2)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw towers
        for t in self.towers:
            spec = self.TOWER_SPECS[t["type"]]
            pos = (int(t["pos"][0]), int(t["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GRID_SIZE // 2, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GRID_SIZE // 2, spec["color"])
            if t["cooldown"] > 0:
                angle = (t["cooldown"] / spec["cooldown"]) * 360
                arc_rect = pygame.Rect(pos[0] - self.GRID_SIZE//2, pos[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.arc(self.screen, (255,255,255), arc_rect, math.radians(90), math.radians(90+angle), 2)

        # Draw zombies
        for z in self.zombies:
            pos = (int(z["pos"][0]), int(z["pos"][1]))
            color = self.COLOR_ZOMBIE_SLOWED if z["slow_timer"] > 0 else self.COLOR_ZOMBIE
            pygame.gfxdraw.box(self.screen, (pos[0]-5, pos[1]-5, 10, 10), color)
            # Health bar
            health_pct = max(0, z["health"] / z["max_health"])
            pygame.draw.rect(self.screen, (50, 50, 50), (pos[0] - 8, pos[1] - 12, 16, 4))
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] - 8, pos[1] - 12, 16 * health_pct, 4))

        # Draw projectiles
        for p in self.projectiles:
            spec = self.TOWER_SPECS[p["type"]]
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, spec["color"])

        # Draw particles
        for particle in self.particles:
            pos = (int(particle["pos"][0]), int(particle["pos"][1]))
            size = int(particle["life"] / particle["max_life"] * particle["size"])
            if size > 0:
                pygame.draw.circle(self.screen, particle["color"], pos, size)

        # Draw cursor
        cursor_world_x = self.cursor_pos[0] * self.GRID_SIZE
        cursor_world_y = self.cursor_pos[1] * self.GRID_SIZE
        cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, (cursor_world_x, cursor_world_y))

        # Draw tower range preview
        spec = self.TOWER_SPECS[self.selected_tower_type]
        center_x = cursor_world_x + self.GRID_SIZE // 2
        center_y = cursor_world_y + self.GRID_SIZE // 2
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, spec["range"], (255, 255, 255, 50))

    def _render_ui(self):
        # Top Left: Wave info
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Top Right: Base Health
        health_text = self.font_m.render(f"Base Health: {int(self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))
        
        # Bottom Left: Money
        money_text = self.font_m.render(f"$ {self.money}", True, self.TOWER_SPECS[0]["color"])
        self.screen.blit(money_text, (10, self.SCREEN_HEIGHT - money_text.get_height() - 10))

        # Bottom Right: Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name_text = self.font_s.render(f"{spec['name']}", True, self.COLOR_TEXT)
        tower_cost_text = self.font_s.render(f"Cost: ${spec['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(tower_name_text, (self.SCREEN_WIDTH - tower_name_text.get_width() - 10, self.SCREEN_HEIGHT - 40))
        self.screen.blit(tower_cost_text, (self.SCREEN_WIDTH - tower_cost_text.get_width() - 10, self.SCREEN_HEIGHT - 20))

        # Center: Score
        score_text = self.font_s.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - score_text.get_height() - 5))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            win_condition = self.wave_number > self.MAX_WAVES
            end_text_str = "VICTORY" if win_condition else "GAME OVER"
            end_text_color = self.COLOR_BASE if win_condition else self.COLOR_ZOMBIE
            
            end_text = self.font_l.render(end_text_str, True, end_text_color)
            final_score_text = self.font_m.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            
            self.screen.blit(end_text, (self.SCREEN_WIDTH//2 - end_text.get_width()//2, self.SCREEN_HEIGHT//2 - 40))
            self.screen.blit(final_score_text, (self.SCREEN_WIDTH//2 - final_score_text.get_width()//2, self.SCREEN_HEIGHT//2 + 10))

    def _create_particles(self, pos, color, count, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Set to True to play manually with keyboard.
    # Requires pygame to be installed with display drivers.
    MANUAL_PLAY = False 
    
    if MANUAL_PLAY:
        pygame.display.set_caption("Tower Defense")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        terminated = False
        while not terminated:
            # Action defaults to NO-OP
            action = [0, 0, 0] 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display screen
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(draw_surface, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit to 30 FPS

    else: # Run with random actions
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Final Info: {info}")
                obs, info = env.reset()

    env.close()