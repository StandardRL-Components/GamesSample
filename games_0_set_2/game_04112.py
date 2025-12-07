
# Generated: 2025-08-28T01:27:17.823829
# Source Brief: brief_04112.md
# Brief Index: 4112

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press Shift to cycle tower types. Press Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this minimalist top-down tower defense game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PATH = (60, 60, 70)
    COLOR_BASE = (60, 180, 75)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_CURSOR_VALID = (255, 255, 25, 150)
    COLOR_CURSOR_INVALID = (255, 0, 0, 150)
    COLOR_TEXT = (245, 245, 245)
    COLOR_MONEY = (255, 215, 0)

    # Game parameters
    GRID_SIZE = 40
    MAX_STEPS = 2000
    TOTAL_ENEMIES = 20
    INITIAL_MONEY = 100
    MONEY_PER_KILL = 25
    
    # Tower definitions: [cost, range, damage, fire_rate (steps), color]
    TOWER_SPECS = {
        0: {"cost": 50, "range": 80, "damage": 10, "fire_rate": 20, "color": (0, 130, 200)}, # Blue: Balanced
        1: {"cost": 75, "range": 120, "damage": 5, "fire_rate": 8, "color": (255, 225, 25)}, # Yellow: Fast, long range
        2: {"cost": 100, "range": 60, "damage": 35, "fire_rate": 45, "color": (145, 30, 180)}, # Purple: Slow, high damage
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Path and grid setup
        self.path = self._define_path()
        self.grid_w = self.width // self.GRID_SIZE
        self.grid_h = self.height // self.GRID_SIZE
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def _define_path(self):
        return [
            (-20, 180), (80, 180), (80, 80), (240, 80), (240, 280),
            (400, 280), (400, 120), (560, 120), (560, 340), (self.width + 20, 340)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.money = self.INITIAL_MONEY
        self.game_over = False
        self.victory = False

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.enemies_to_spawn = self.TOTAL_ENEMIES
        self.enemies_defeated = 0
        self.base_health = 1 # Game ends if 1 enemy reaches base
        
        self.spawn_cooldown_max = 60 # Steps between spawns
        self.spawn_timer = self.spawn_cooldown_max

        self.cursor_grid_pos = [self.grid_w // 2, self.grid_h // 2]
        self.selected_tower_type = 0
        self.last_shift_held = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Start with a small time penalty
        reward = -0.01

        # Handle player actions
        reward += self._handle_actions(movement, space_held, shift_held)

        # Update game logic
        self._spawn_enemies()
        step_rewards = self._update_entities()
        reward += step_rewards

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Loss penalty
        elif self.enemies_defeated >= self.TOTAL_ENEMIES and not self.enemies:
            self.game_over = True
            self.victory = True
            terminated = True
            reward += 100 # Victory bonus
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_actions(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_grid_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1  # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1  # Right
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.grid_w - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.grid_h - 1)

        # Cycle tower type (on press, not hold)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_held = shift_held

        # Place tower
        if space_held:
            cursor_pos = (
                (self.cursor_grid_pos[0] + 0.5) * self.GRID_SIZE,
                (self.cursor_grid_pos[1] + 0.5) * self.GRID_SIZE
            )
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self._is_valid_placement(cursor_pos, spec["cost"]):
                self.money -= spec["cost"]
                self.towers.append({
                    "pos": cursor_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                })
                # Sound: build_tower.wav
                return 0.5  # Small reward for a valid strategic decision
        return 0

    def _is_valid_placement(self, pos, cost):
        if self.money < cost:
            return False
        
        # Check proximity to path
        for i in range(len(self.path) - 1):
            p1 = pygame.Vector2(self.path[i])
            p2 = pygame.Vector2(self.path[i+1])
            line_rect = pygame.Rect(min(p1.x, p2.x), min(p1.y, p2.y), abs(p1.x-p2.x), abs(p1.y-p2.y))
            if line_rect.inflate(self.GRID_SIZE, self.GRID_SIZE).collidepoint(pygame.Vector2(pos)):
                 return False

        # Check proximity to other towers
        for tower in self.towers:
            dist = math.hypot(pos[0] - tower["pos"][0], pos[1] - tower["pos"][1])
            if dist < self.GRID_SIZE:
                return False
        
        return True

    def _spawn_enemies(self):
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append({
                    "pos": pygame.Vector2(self.path[0]),
                    "health": 50 + self.enemies_defeated * 2, # Enemies get tougher
                    "max_health": 50 + self.enemies_defeated * 2,
                    "speed": 1.0 + self.enemies_defeated * 0.05,
                    "waypoint_idx": 1,
                    "dist_on_path": 0,
                })
                self.enemies_to_spawn -= 1
                
                # Difficulty scaling: spawn faster as more enemies are defeated
                spawn_reduction = (self.enemies_defeated // 5) * 5
                self.spawn_timer = max(15, self.spawn_cooldown_max - spawn_reduction)
                # Sound: enemy_spawn.wav

    def _update_entities(self):
        reward = 0
        
        # Update towers (targeting and shooting)
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
            else:
                spec = self.TOWER_SPECS[tower["type"]]
                target = self._find_target(tower)
                if target:
                    tower["cooldown"] = spec["fire_rate"]
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower["pos"]),
                        "target_enemy": target,
                        "speed": 8,
                        "damage": spec["damage"],
                    })
                    # Sound: shoot.wav

        # Update projectiles
        for proj in self.projectiles[:]:
            if proj["target_enemy"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj["target_enemy"]["pos"]
            direction = (target_pos - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]
            
            if proj["pos"].distance_to(target_pos) < 5:
                proj["target_enemy"]["health"] -= proj["damage"]
                reward += 0.1 # Reward for hitting
                self._create_particles(proj["pos"], self.COLOR_PROJECTILE, 5)
                self.projectiles.remove(proj)
                # Sound: hit_enemy.wav

        # Update enemies
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                self.score += 10
                self.money += self.MONEY_PER_KILL
                reward += 1.0 # Reward for kill
                self.enemies_defeated += 1
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15)
                self.enemies.remove(enemy)
                # Sound: enemy_die.wav
                continue

            # Movement
            if enemy["waypoint_idx"] < len(self.path):
                target_pos = pygame.Vector2(self.path[enemy["waypoint_idx"]])
                direction = (target_pos - enemy["pos"])
                dist = direction.length()
                
                if dist < enemy["speed"]:
                    enemy["waypoint_idx"] += 1
                    enemy["dist_on_path"] += dist
                else:
                    move_vec = direction.normalize() * enemy["speed"]
                    enemy["pos"] += move_vec
                    enemy["dist_on_path"] += move_vec.length()
            else: # Reached end of path
                self.base_health -= 1
                self.enemies.remove(enemy)
                # Sound: base_damage.wav
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return reward

    def _find_target(self, tower):
        spec = self.TOWER_SPECS[tower["type"]]
        tower_pos = pygame.Vector2(tower["pos"])
        
        valid_targets = [e for e in self.enemies if tower_pos.distance_to(e["pos"]) <= spec["range"]]

        if not valid_targets:
            return None
        
        # Target enemy furthest along the path
        return max(valid_targets, key=lambda e: e["dist_on_path"])

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path and base
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 5)
        base_pos = self.path[-2]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos[0]-10, base_pos[1]-10, 20, 20))

        # Draw towers and their ranges (if cursor is over them)
        cursor_world_pos = ((self.cursor_grid_pos[0] + 0.5) * self.GRID_SIZE, (self.cursor_grid_pos[1] + 0.5) * self.GRID_SIZE)
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos_int = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.draw.rect(self.screen, spec["color"], (pos_int[0] - 15, pos_int[1] - 15, 30, 30))
            if math.hypot(cursor_world_pos[0] - pos_int[0], cursor_world_pos[1] - pos_int[1]) < self.GRID_SIZE / 2:
                 pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], spec["range"], (*spec["color"], 100))

        # Draw enemies with health bars
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)
            health_ratio = max(0, enemy["health"] / enemy["max_health"])
            pygame.draw.rect(self.screen, (255,0,0), (pos_int[0]-10, pos_int[1]-15, 20, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos_int[0]-10, pos_int[1]-15, 20 * health_ratio, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            alpha = max(0, min(255, int(p['life'] * 15)))
            color = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 2, color)
            except TypeError: # Color might have invalid alpha
                pass

        # Draw placement cursor
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_valid = self._is_valid_placement(cursor_world_pos, spec["cost"])
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.aacircle(self.screen, int(cursor_world_pos[0]), int(cursor_world_pos[1]), spec["range"], cursor_color)
        pygame.draw.rect(self.screen, cursor_color[:-1], (cursor_world_pos[0] - 15, cursor_world_pos[1] - 15, 30, 30), 2)

    def _render_ui(self):
        # Score, Money, Enemies
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        money_text = self.font_large.render(f"${self.money}", True, self.COLOR_MONEY)
        self.screen.blit(money_text, (10, 40))
        enemies_text = self.font_small.render(f"Enemies: {self.enemies_to_spawn + len(self.enemies)}/{self.TOTAL_ENEMIES}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.width - enemies_text.get_width() - 10, 10))
        
        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = self.font_small.render(f"Cost: ${spec['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (self.width - tower_info_text.get_width() - 10, self.height - 30))
        pygame.draw.rect(self.screen, spec["color"], (self.width - 100, self.height - 60, 20, 20))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "enemies_defeated": self.enemies_defeated,
            "enemies_remaining": self.enemies_to_spawn + len(self.enemies),
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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            # For manual play, we reset after a delay to see the end screen
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()