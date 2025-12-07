
# Generated: 2025-08-28T06:41:36.705035
# Source Brief: brief_03009.md
# Brief Index: 3009

        
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
        "Controls: ↑↓←→ to move the cursor. Space to place a tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers in a top-down tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (45, 50, 66)
    COLOR_BASE = (66, 179, 116)
    COLOR_ENEMY = (219, 79, 79)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (35, 38, 48)
    COLOR_INVALID = (255, 0, 0, 100)
    
    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 15000 
    FPS = 30
    BASE_HEALTH_START = 100
    STARTING_RESOURCES = 150
    MAX_WAVES = 10
    REWARD_ENEMY_KILLED = 0.5
    REWARD_WAVE_CLEARED = 10
    REWARD_VICTORY = 100
    REWARD_DEFEAT = -100
    CURSOR_SPEED = 10

    # Tower Specs
    TOWER_SPECS = {
        0: {
            "name": "Gatling",
            "cost": 50,
            "range": 80,
            "damage": 3,
            "fire_rate": 0.2, # seconds per shot
            "color": (255, 215, 0),
            "proj_speed": 8,
            "proj_color": (255, 255, 100)
        },
        1: {
            "name": "Cannon",
            "cost": 120,
            "range": 120,
            "damage": 25,
            "fire_rate": 1.5,
            "color": (0, 191, 255),
            "proj_speed": 6,
            "proj_color": (173, 216, 230)
        }
    }

    # Enemy Specs
    ENEMY_SPECS = {
        "normal": {"health": 20, "speed": 1.0, "size": 12, "reward": 10},
        "fast": {"health": 15, "speed": 1.8, "size": 10, "reward": 15},
        "heavy": {"health": 80, "speed": 0.7, "size": 16, "reward": 25}
    }

    # Wave Definitions
    WAVE_DEFINITIONS = [
        {"normal": 5},
        {"normal": 10},
        {"normal": 10, "fast": 3},
        {"normal": 15, "fast": 5},
        {"heavy": 2, "normal": 10},
        {"fast": 15, "heavy": 2},
        {"heavy": 5, "fast": 10},
        {"normal": 20, "heavy": 5},
        {"fast": 20, "heavy": 8},
        {"normal": 15, "fast": 15, "heavy": 10}
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        self.path_waypoints = [
            pygame.Vector2(-20, 150), pygame.Vector2(100, 150),
            pygame.Vector2(100, 300), pygame.Vector2(250, 300),
            pygame.Vector2(250, 100), pygame.Vector2(450, 100),
            pygame.Vector2(450, 250), pygame.Vector2(self.WIDTH + 20, 250)
        ]
        self.base_pos = pygame.Vector2(self.WIDTH - 40, 250)
        self.base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.selected_tower_type = 0
        self.was_space_pressed = False
        self.was_shift_pressed = False
        self.wave_in_progress = False
        self.time_to_next_wave = 0
        self.wave_spawn_list = []
        self.spawn_timer = 0
        self.victory = False

        self.reset()
        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = self.BASE_HEALTH_START
        self.resources = self.STARTING_RESOURCES
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.selected_tower_type = 0
        self.was_space_pressed = False
        self.was_shift_pressed = False
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        self._handle_input(movement, space_action, shift_action)
        
        # --- Place Tower Action ---
        if self._place_tower():
            # Placeholder for sound effect
            # sfx: place_tower.wav
            pass

        # --- Update Game Logic ---
        self._update_spawner()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Wave Management ---
        if not self.wave_in_progress and not self.enemies:
            if self.wave_number >= self.MAX_WAVES:
                self.victory = True
                self.game_over = True
                reward += self.REWARD_VICTORY
            else:
                self.time_to_next_wave -= 1 / self.FPS
                if self.time_to_next_wave <= 0:
                    reward += self.REWARD_WAVE_CLEARED
                    self._start_next_wave()

        self.steps += 1
        self.score += reward
        
        # --- Check Termination ---
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            reward += self.REWARD_DEFEAT
            # sfx: game_over.wav
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_action, shift_action):
        # Movement
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Cycle tower type (on press)
        is_shift_pressed = shift_action == 1
        if is_shift_pressed and not self.was_shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_cycle.wav
        self.was_shift_pressed = is_shift_pressed
        
        # Prepare for tower placement (on press)
        self.is_space_pressed = space_action == 1

    def _is_valid_placement(self, pos):
        # Check resources
        if self.resources < self.TOWER_SPECS[self.selected_tower_type]["cost"]:
            return False
        # Check proximity to path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            # Simplified line-point distance check
            rect = pygame.Rect(min(p1.x, p2.x) - 20, min(p1.y, p2.y) - 20, abs(p1.x - p2.x) + 40, abs(p1.y - p2.y) + 40)
            if rect.collidepoint(pos):
                return False
        # Check proximity to other towers
        for tower in self.towers:
            if pos.distance_to(tower["pos"]) < 30:
                return False
        # Check proximity to base
        if pos.distance_to(self.base_pos) < 40:
            return False
        return True

    def _place_tower(self):
        if self.is_space_pressed and not self.was_space_pressed:
            self.was_space_pressed = True
            if self._is_valid_placement(self.cursor_pos):
                spec = self.TOWER_SPECS[self.selected_tower_type]
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "type": self.selected_tower_type,
                    "cooldown": 0
                })
                return True
        if not self.is_space_pressed:
            self.was_space_pressed = False
        return False

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES: return

        self.wave_in_progress = True
        self.time_to_next_wave = 5 # seconds
        self.spawn_timer = 0
        self.wave_spawn_list = []
        
        wave_data = self.WAVE_DEFINITIONS[self.wave_number - 1]
        for enemy_type, count in wave_data.items():
            for _ in range(count):
                self.wave_spawn_list.append(enemy_type)
        random.shuffle(self.wave_spawn_list)

    def _update_spawner(self):
        if self.wave_in_progress and self.wave_spawn_list:
            self.spawn_timer -= 1 / self.FPS
            if self.spawn_timer <= 0:
                enemy_type = self.wave_spawn_list.pop(0)
                self._spawn_enemy(enemy_type)
                self.spawn_timer = 0.5 # Time between spawns in a wave
                if not self.wave_spawn_list:
                    self.wave_in_progress = False

    def _spawn_enemy(self, enemy_type):
        spec = self.ENEMY_SPECS[enemy_type]
        difficulty_mod = (1.05 ** (self.wave_number - 1))
        
        self.enemies.append({
            "pos": self.path_waypoints[0].copy(),
            "type": enemy_type,
            "health": spec["health"] * difficulty_mod,
            "max_health": spec["health"] * difficulty_mod,
            "speed": spec["speed"],
            "size": spec["size"],
            "reward": spec["reward"],
            "waypoint_index": 1,
            "distance_traveled": 0,
        })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] -= 1 / self.FPS
            if tower["cooldown"] <= 0:
                # Find target
                target = None
                max_dist = -1
                for enemy in self.enemies:
                    dist = tower["pos"].distance_to(enemy["pos"])
                    if dist <= spec["range"] and enemy["distance_traveled"] > max_dist:
                        max_dist = enemy["distance_traveled"]
                        target = enemy
                
                if target:
                    # Fire projectile
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": target,
                        "type": tower["type"]
                    })
                    tower["cooldown"] = spec["fire_rate"]
                    # sfx: shoot_gatling.wav or shoot_cannon.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            spec = self.TOWER_SPECS[proj["type"]]
            target = proj["target"]
            
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = (target["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * spec["proj_speed"]
            
            if proj["pos"].distance_to(target["pos"]) < target["size"] / 2:
                target["health"] -= spec["damage"]
                self.projectiles.remove(proj)
                # sfx: hit_enemy.wav
                for _ in range(5):
                    self.particles.append(self._create_particle(proj["pos"], spec["proj_color"]))

                if target["health"] <= 0:
                    self.resources += target["reward"]
                    reward += self.REWARD_ENEMY_KILLED
                    for _ in range(20):
                        self.particles.append(self._create_particle(target["pos"], self.COLOR_ENEMY))
                    self.enemies.remove(target)
                    # sfx: enemy_explode.wav
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["waypoint_index"] >= len(self.path_waypoints):
                self.base_health -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                continue
                
            target_pos = self.path_waypoints[enemy["waypoint_index"]]
            direction = (target_pos - enemy["pos"])
            
            if direction.length() < enemy["speed"]:
                enemy["pos"] = target_pos.copy()
                enemy["waypoint_index"] += 1
            else:
                move_vec = direction.normalize() * enemy["speed"]
                enemy["pos"] += move_vec
                enemy["distance_traveled"] += move_vec.length()
        return reward

    def _create_particle(self, pos, color):
        return {
            "pos": pos.copy(),
            "vel": pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
            "lifespan": random.uniform(0.3, 0.8),
            "color": color
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1 / self.FPS
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.path_waypoints) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_waypoints[i], self.path_waypoints[i+1], 30)

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        
        # Draw towers and ranges
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos_int = (int(tower["pos"].x), int(tower["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, self.COLOR_TEXT)
            if tower["pos"].distance_to(self.cursor_pos) < 15:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(spec["range"]), (255,255,255,50))


        # Draw enemies and health bars
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            size = int(enemy["size"])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos_int[0] - size/2, pos_int[1] - size/2, size, size))
            
            # Health bar
            health_ratio = max(0, enemy["health"] / enemy["max_health"])
            bar_width = size * 1.2
            pygame.draw.rect(self.screen, (80,0,0), (pos_int[0] - bar_width/2, pos_int[1] - size, bar_width, 5))
            pygame.draw.rect(self.screen, (0,200,0), (pos_int[0] - bar_width/2, pos_int[1] - size, bar_width * health_ratio, 5))

        # Draw projectiles
        for proj in self.projectiles:
            spec = self.TOWER_SPECS[proj["type"]]
            pos_int = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, spec["proj_color"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["lifespan"] * 255 * 2)))
            color = (*p["color"], alpha)
            s = pygame.Surface((3,3), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0,0,3,3))
            self.screen.blit(s, (int(p["pos"].x), int(p["pos"].y)))
            
        # Draw cursor
        cursor_spec = self.TOWER_SPECS[self.selected_tower_type]
        cursor_pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        is_valid = self._is_valid_placement(self.cursor_pos)
        
        range_color = (255, 255, 255, 50) if is_valid else (255, 0, 0, 50)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], int(cursor_spec["range"]), range_color)
        
        tower_color = cursor_spec["color"] if is_valid else self.COLOR_INVALID
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, tower_color)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, self.COLOR_TEXT)


    def _render_ui(self):
        # Top-left UI: Wave info
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        wave_surf = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (10, 10))
        
        if not self.wave_in_progress and not self.victory:
            next_wave_text = f"Next wave in: {int(self.time_to_next_wave)}s"
            next_wave_surf = self.font_small.render(next_wave_text, True, self.COLOR_TEXT)
            self.screen.blit(next_wave_surf, (10, 35))

        # Top-right UI: Base Health
        health_text = f"Base Health: {int(self.base_health)}"
        health_surf = self.font_large.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(health_surf, (self.WIDTH - health_surf.get_width() - 10, 10))
        
        # Bottom-left UI: Resources and Tower selection
        res_text = f"Resources: ${self.resources}"
        res_surf = self.font_large.render(res_text, True, (255, 215, 0))
        self.screen.blit(res_surf, (10, self.HEIGHT - 40))
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        sel_text = f"Selected: {spec['name']} (${spec['cost']})"
        sel_surf = self.font_small.render(sel_text, True, self.COLOR_TEXT)
        self.screen.blit(sel_surf, (10, self.HEIGHT - 60))

        # Game Over / Victory screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            
            end_text_surf = self.font_huge.render(message, True, color)
            pos_x = self.WIDTH/2 - end_text_surf.get_width()/2
            pos_y = self.HEIGHT/2 - end_text_surf.get_height()/2
            self.screen.blit(end_text_surf, (pos_x, pos_y))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "enemies_remaining": len(self.enemies) + len(self.wave_spawn_list)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to render to screen
    render_mode = "human" # "rgb_array" or "human"

    if render_mode == "human":
        GameEnv.metadata["render_modes"].append("human")
        GameEnv.render = lambda self, mode="human": pygame.display.get_surface().blit(self.screen, (0, 0))
        pygame.display.set_caption("Tower Defense")
        pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control ---
    # To play manually, a mapping from keyboard to action is needed.
    # This is a simplified example; a real human player would need a more robust wrapper.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    action[0] = move_action
                    break
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
        else:
            # Agent control (random actions)
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            time.sleep(2)
            obs, info = env.reset()
            terminated = False

    env.close()