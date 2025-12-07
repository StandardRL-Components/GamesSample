
# Generated: 2025-08-28T02:52:16.661335
# Source Brief: brief_01839.md
# Brief Index: 1839

        
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
        "Controls: Arrows to move cursor. Shift to cycle tower types. Space to place selected tower."
    )

    game_description = (
        "Defend your base from relentless waves of zombies by strategically placing towers."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 18000 # 10 minutes at 30fps
        self.MAX_WAVES = 30

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (60, 60, 70)
        self.COLOR_ZONE = (45, 45, 55)
        self.COLOR_ZONE_HIGHLIGHT = (200, 200, 100)
        self.COLOR_BASE = (0, 150, 0)
        self.COLOR_BASE_DAMAGED = (150, 150, 0)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_HEALTH_GREEN = (0, 255, 0)
        self.COLOR_HEALTH_RED = (255, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_HEADER = (255, 255, 255)
        self.COLOR_TEXT_MONEY = (255, 223, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 28, bold=True)

        # --- Game Configuration ---
        self.PATH = [
            (-20, 100), (80, 100), (80, 300), (240, 300), (240, 100),
            (400, 100), (400, 300), (560, 300), (560, 100), (self.WIDTH + 20, 100)
        ]
        self.BASE_POS = (self.WIDTH - 40, 80)
        self.BASE_SIZE = (40, 40)
        self.MAX_BASE_HEALTH = 100

        self.TOWER_ZONES = []
        for y in [40, 160, 220, 340]:
            for x in range(40, self.WIDTH, 80):
                self.TOWER_ZONES.append((x, y))

        self.TOWER_SPECS = [
            {"name": "Gun", "cost": 100, "range": 70, "damage": 2, "cooldown": 20, "color": (0, 150, 255), "proj_speed": 8},
            {"name": "Cannon", "cost": 250, "range": 90, "damage": 10, "cooldown": 60, "color": (150, 100, 255), "proj_speed": 6},
            {"name": "Sniper", "cost": 400, "range": 200, "damage": 25, "cooldown": 100, "color": (255, 255, 0), "proj_speed": 20},
            {"name": "Slower", "cost": 150, "range": 60, "damage": 0.5, "cooldown": 30, "color": (0, 255, 255), "proj_speed": 7, "slow_factor": 0.5},
            {"name": "Flamer", "cost": 300, "range": 50, "damage": 0.5, "cooldown": 5, "color": (255, 120, 0), "proj_speed": 5, "is_flame": True}
        ]

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.zombies_to_spawn = []
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = (0, 0) # grid index, not pixels
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_timer = 0
        self.wave_in_progress = False

        self.reset()
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.MAX_BASE_HEALTH
        self.resources = 250
        self.wave_number = 0
        self.zombies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.zombies_to_spawn.clear()
        self.cursor_pos = (len(self.TOWER_ZONES) // 2)
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_in_progress = False
        self.wave_timer = 150 # 5 seconds before first wave

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001 # Small penalty for time passing
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_wave_logic()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()

        # --- Check Termination ---
        terminated = False
        if self.base_health <= 0:
            reward = -100
            terminated = True
            self.game_over = True
        elif self.wave_number > self.MAX_WAVES:
            reward = 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_pos = (self.cursor_pos - 8) % len(self.TOWER_ZONES)
        elif movement == 2: self.cursor_pos = (self.cursor_pos + 8) % len(self.TOWER_ZONES)
        elif movement == 3: self.cursor_pos = (self.cursor_pos - 1) % len(self.TOWER_ZONES)
        elif movement == 4: self.cursor_pos = (self.cursor_pos + 1) % len(self.TOWER_ZONES)

        # Shift (cycle tower): trigger on press
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_held = shift_held

        # Space (place tower): trigger on press
        if space_held and not self.last_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"]:
                pos = self.TOWER_ZONES[self.cursor_pos]
                is_occupied = any(t['pos'] == pos for t in self.towers)
                if not is_occupied:
                    self.resources -= spec["cost"]
                    self.towers.append({
                        "pos": pos,
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                        "angle": -math.pi / 2
                    })
                    # Sound: place_tower.wav
        self.last_space_held = space_held
        return 0

    def _update_wave_logic(self):
        if self.wave_in_progress:
            if not self.zombies and not self.zombies_to_spawn:
                self.wave_in_progress = False
                self.wave_timer = 300 # 10s between waves
                return 1.0 # Wave clear reward
        else:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                if self.wave_number <= self.MAX_WAVES:
                    self._start_new_wave()
        return 0.0

    def _start_new_wave(self):
        self.wave_in_progress = True
        num_zombies = 5 + self.wave_number
        base_health = 10 + (self.wave_number // 2)
        speed = 1.0 + (self.wave_number // 5) * 0.1
        
        self.zombies_to_spawn = [
            {"health": base_health, "max_health": base_health, "speed": speed, "slow_timer": 0}
            for _ in range(num_zombies)
        ]
        self.wave_timer = 15 # Spawn interval

    def _spawn_zombie(self):
        if self.wave_in_progress and self.zombies_to_spawn:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                spec = self.zombies_to_spawn.pop(0)
                self.zombies.append({
                    "pos": pygame.Vector2(self.PATH[0]),
                    "health": spec["health"],
                    "max_health": spec["max_health"],
                    "speed": spec["speed"],
                    "slow_timer": spec["slow_timer"],
                    "path_index": 1,
                })
                self.wave_timer = max(5, 20 - self.wave_number // 2) # Spawn faster in later waves

    def _update_zombies(self):
        reward = 0
        self._spawn_zombie()
        
        for z in reversed(self.zombies):
            if z["slow_timer"] > 0:
                z["slow_timer"] -= 1
                current_speed = z["speed"] * 0.5
            else:
                current_speed = z["speed"]

            if z["path_index"] < len(self.PATH):
                target_pos = pygame.Vector2(self.PATH[z["path_index"]])
                direction = (target_pos - z["pos"]).normalize()
                z["pos"] += direction * current_speed

                if z["pos"].distance_to(target_pos) < current_speed:
                    z["path_index"] += 1
            else: # Reached base
                self.base_health = max(0, self.base_health - z["health"] / 2)
                self.zombies.remove(z)
                # Sound: base_damage.wav
        return reward

    def _update_towers(self):
        for t in self.towers:
            spec = self.TOWER_SPECS[t["type"]]
            if t["cooldown"] > 0:
                t["cooldown"] -= 1
                continue

            target = None
            min_dist = spec["range"]
            for z in self.zombies:
                dist = pygame.Vector2(t["pos"]).distance_to(z["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = z
            
            if target:
                t["cooldown"] = spec["cooldown"]
                target_pos = target["pos"]
                t["angle"] = math.atan2(target_pos.y - t["pos"][1], target_pos.x - t["pos"][0])
                
                self.projectiles.append({
                    "pos": pygame.Vector2(t["pos"]),
                    "type": t["type"],
                    "target_pos": target_pos,
                    "angle": t["angle"]
                })
                # Sound: shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in reversed(self.projectiles):
            spec = self.TOWER_SPECS[p["type"]]
            p["pos"] += pygame.Vector2(math.cos(p["angle"]), math.sin(p["angle"])) * spec["proj_speed"]

            if not self.screen.get_rect().collidepoint(p["pos"]):
                self.projectiles.remove(p)
                continue

            for z in reversed(self.zombies):
                if p["pos"].distance_to(z["pos"]) < 10:
                    z["health"] -= spec["damage"]
                    if "slow_factor" in spec:
                        z["slow_timer"] = 60 # 2 seconds
                    
                    self._create_particles(p["pos"], spec["color"])
                    
                    if z["health"] <= 0:
                        self.resources += 5 + self.wave_number // 5
                        reward += 0.1
                        self.zombies.remove(z)
                        # Sound: zombie_die.wav
                    
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    break
        return reward

    def _create_particles(self, pos, color, count=5):
        for _ in range(count):
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                "life": 15,
                "color": color
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH, 40)
        pygame.draw.lines(self.screen, (80,80,90), False, self.PATH, 2)

        # Draw tower zones
        for i, pos in enumerate(self.TOWER_ZONES):
            color = self.COLOR_ZONE_HIGHLIGHT if i == self.cursor_pos else self.COLOR_ZONE
            pygame.gfxdraw.box(self.screen, pygame.Rect(pos[0]-15, pos[1]-15, 30, 30), (*color, 100))
            pygame.gfxdraw.rectangle(self.screen, pygame.Rect(pos[0]-15, pos[1]-15, 30, 30), (*color, 150))

        # Draw base
        base_rect = pygame.Rect(self.BASE_POS, self.BASE_SIZE)
        health_ratio = self.base_health / self.MAX_BASE_HEALTH
        base_color = self.COLOR_BASE if health_ratio > 0.5 else self.COLOR_BASE_DAMAGED
        pygame.draw.rect(self.screen, base_color, base_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (base_rect.x, base_rect.y - 10, base_rect.w * health_ratio, 5))

        # Draw towers
        for t in self.towers:
            spec = self.TOWER_SPECS[t["type"]]
            pos = (int(t["pos"][0]), int(t["pos"][1]))
            pygame.draw.circle(self.screen, (20,20,20), pos, 14)
            pygame.draw.circle(self.screen, spec["color"], pos, 12)
            # Draw barrel
            end_x = pos[0] + 15 * math.cos(t["angle"])
            end_y = pos[1] + 15 * math.sin(t["angle"])
            pygame.draw.line(self.screen, spec["color"], pos, (int(end_x), int(end_y)), 4)

        # Draw zombies
        for z in self.zombies:
            pos = (int(z["pos"].x), int(z["pos"].y))
            size = 8
            z_rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
            color = (150, 0, 200) if z["slow_timer"] > 0 else self.COLOR_ZOMBIE
            pygame.draw.rect(self.screen, color, z_rect)
            # Health bar
            health_w = 12
            health_ratio = z["health"] / z["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos[0] - health_w//2, pos[1] - 12, health_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0] - health_w//2, pos[1] - 12, health_w * health_ratio, 3))

        # Draw projectiles
        for p in self.projectiles:
            spec = self.TOWER_SPECS[p["type"]]
            pos = (int(p["pos"].x), int(p["pos"].y))
            if spec.get("is_flame", False):
                pygame.draw.circle(self.screen, spec["color"], pos, random.randint(3,6))
            else:
                pygame.draw.circle(self.screen, spec["color"], pos, 3)
                pygame.draw.circle(self.screen, (255,255,255), pos, 1)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 15))))
            size = int(p["life"] / 4)
            color = (*p["color"], alpha)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, color)

    def _render_ui(self):
        # Top Left: Wave Info
        wave_text = self.font_medium.render(f"WAVE {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT_HEADER)
        self.screen.blit(wave_text, (10, 10))
        if not self.wave_in_progress and self.wave_number < self.MAX_WAVES:
            next_wave_in = self.font_small.render(f"Next wave in {self.wave_timer / self.FPS:.1f}s", True, self.COLOR_TEXT)
            self.screen.blit(next_wave_in, (10, 35))
        elif self.wave_in_progress:
            zombies_left = len(self.zombies) + len(self.zombies_to_spawn)
            zombies_text = self.font_small.render(f"Zombies: {zombies_left}", True, self.COLOR_ZOMBIE)
            self.screen.blit(zombies_text, (10, 35))

        # Top Right: Base Health
        health_text = self.font_medium.render(f"BASE HP: {int(self.base_health)}", True, self.COLOR_HEALTH_GREEN)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        
        # Bottom Left: Resources
        resource_text = self.font_large.render(f"${int(self.resources)}", True, self.COLOR_TEXT_MONEY)
        self.screen.blit(resource_text, (10, self.HEIGHT - resource_text.get_height() - 10))

        # Bottom Right: Tower Selection
        spec = self.TOWER_SPECS[self.selected_tower_type]
        name_text = self.font_medium.render(f"{spec['name']}", True, spec["color"])
        cost_text = self.font_small.render(f"Cost: ${spec['cost']}", True, self.COLOR_TEXT)
        damage_text = self.font_small.render(f"Dmg: {spec['damage']} Rng: {spec['range']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (self.WIDTH - name_text.get_width() - 10, self.HEIGHT - 60))
        self.screen.blit(cost_text, (self.WIDTH - cost_text.get_width() - 10, self.HEIGHT - 40))
        self.screen.blit(damage_text, (self.WIDTH - damage_text.get_width() - 10, self.HEIGHT - 25))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text = "VICTORY!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            color = (0, 255, 0) if self.wave_number > self.MAX_WAVES else (255, 0, 0)
            
            end_text_surf = self.font_large.render(result_text, True, color)
            end_text_rect = end_text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text_surf, end_text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "zombies_alive": len(self.zombies),
            "towers_placed": len(self.towers)
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # Pygame window for human interaction
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)
    
    while not done:
        # --- Event Handling for Manual Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        # Get key states for continuous actions
        keys = pygame.key.get_pressed()
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Create the action tuple
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Render to the Display Window ---
        # The observation 'obs' is the rendered game frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Cap the Frame Rate ---
        clock.tick(env.FPS)
        
    env.close()
    print("Game Over. Final Info:", info)