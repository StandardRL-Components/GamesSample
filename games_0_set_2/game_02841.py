
# Generated: 2025-08-28T06:08:34.575731
# Source Brief: brief_02841.md
# Brief Index: 2841

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in the direction you are facing. "
        "Explore the dungeon, defeat enemies, and collect gold to find the exit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down roguelike adventure. Explore procedurally generated dungeons, fight monsters, "
        "and collect treasure. Reach the exit on the third level to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 5000
        self.WIN_LEVEL = 3

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (50, 45, 40)
        self.COLOR_WALL = (80, 70, 60)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_EXIT = (139, 69, 19)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)
        except IOError:
            # Fallback if default font is not found (e.g., in some minimal environments)
            self.font_small = pygame.font.SysFont("sans", 24)
            self.font_large = pygame.font.SysFont("sans", 32)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = 0
        self.player_max_health = 100
        self.player_facing = (0, 1) # (dx, dy)
        self.current_level = 0
        self.gold_collected = 0
        self.dungeon_map = None
        self.enemies = []
        self.gold_piles = []
        self.exit_pos = None
        self.particles = []
        self.steps = 0
        self.game_over = False

        # This will be properly initialized in reset()
        self.np_random = None

        # Validate implementation
        # self.validate_implementation() # Commented out for submission, but used for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.gold_collected = 0
        self.game_over = False
        self.current_level = 1
        self.player_health = self.player_max_health
        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # 1. Create a grid full of walls
        self.dungeon_map = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT)) # 1 = wall, 0 = floor

        # 2. Carve out a path using a random walk
        path_len = self.GRID_WIDTH * self.GRID_HEIGHT // 4
        x, y = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.dungeon_map[x, y] = 0
        
        path = [(x, y)]
        for _ in range(path_len):
            # Get valid neighbors (within bounds and not corner-cutting)
            pot_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1:
                    pot_moves.append((dx, dy))
            
            if not pot_moves: break
            dx, dy = self.np_random.choice(pot_moves, axis=0)
            
            # Walk two steps to create thicker corridors
            for _ in range(2):
                x, y = x + dx, y + dy
                if 0 < x < self.GRID_WIDTH - 1 and 0 < y < self.GRID_HEIGHT - 1:
                    self.dungeon_map[x, y] = 0
                    if (x, y) not in path:
                        path.append((x, y))
                else:
                    break
        
        # 3. Place player, exit, enemies, and gold
        valid_spawn_points = np.argwhere(self.dungeon_map == 0).tolist()
        self.np_random.shuffle(valid_spawn_points)

        self.player_pos = tuple(valid_spawn_points.pop())
        self.player_facing = (0, 1) # Reset facing direction
        
        # Ensure exit is far from player
        far_point_idx = int(len(valid_spawn_points) * 0.9)
        self.exit_pos = tuple(valid_spawn_points.pop(far_point_idx))
        
        # Place enemies
        self.enemies = []
        num_enemies = self.current_level
        enemy_health = 20 + (self.current_level - 1) * 5
        for _ in range(num_enemies):
            if not valid_spawn_points: break
            pos = tuple(valid_spawn_points.pop())
            patrol_end = tuple(valid_spawn_points.pop()) if valid_spawn_points else pos
            self.enemies.append({
                "pos": pos,
                "health": enemy_health,
                "max_health": enemy_health,
                "patrol_start": pos,
                "patrol_target": patrol_end,
                "hit_timer": 0
            })

        # Place gold
        self.gold_piles = []
        num_gold_piles = self.np_random.integers(5, 10)
        for _ in range(num_gold_piles):
            if not valid_spawn_points: break
            pos = tuple(valid_spawn_points.pop())
            value = self.np_random.choice([10, 25, 50], p=[0.6, 0.3, 0.1])
            self.gold_piles.append({"pos": pos, "value": value})
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.02  # Cost of living
        terminated = False

        # --- Player Action Phase ---
        # 1. Movement
        if movement in [1, 2, 3, 4]:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.player_facing = (dx, dy)
            
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            
            # Wall collision check
            if self.dungeon_map[new_x, new_y] == 0:
                self.player_pos = (new_x, new_y)

        # 2. Attack
        if space_held == 1:
            # sfx: player_attack_swoosh.wav
            attack_x = self.player_pos[0] + self.player_facing[0]
            attack_y = self.player_pos[1] + self.player_facing[1]
            
            self._create_particles(1, (attack_x, attack_y), self.COLOR_WHITE, 1.0, 5, 0.2)

            for enemy in self.enemies:
                if enemy["pos"] == (attack_x, attack_y):
                    enemy["health"] -= 10
                    enemy["hit_timer"] = 5 # Flash for 5 frames/turns
                    # sfx: enemy_hit.wav
                    if enemy["health"] <= 0:
                        # sfx: enemy_die.wav
                        reward += 1.0
                        self._create_particles(20, enemy["pos"], self.COLOR_ENEMY, 2.0, 10, 0.5)

            self.enemies = [e for e in self.enemies if e["health"] > 0]
        
        # --- Interaction Phase ---
        # 1. Gold collection
        for gold in self.gold_piles:
            if self.player_pos == gold["pos"]:
                # sfx: gold_collect.wav
                self.gold_collected += gold["value"]
                reward += gold["value"] * 0.1
                self.gold_piles.remove(gold)
                self._create_particles(10, self.player_pos, self.COLOR_GOLD, 1.0, 8, 0.3)
                break
        
        # 2. Exit level
        if self.player_pos == self.exit_pos:
            if self.current_level == self.WIN_LEVEL:
                # sfx: game_win.wav
                reward = 100.0
                terminated = True
                self.game_over = True
            else:
                # sfx: level_up.wav
                self.current_level += 1
                self._generate_level()
                # No extra reward for intermediate levels to encourage finishing

        # --- Enemy Action Phase ---
        for enemy in self.enemies:
            # Simple patrol AI
            target = enemy["patrol_target"]
            if enemy["pos"] == target:
                # Swap target
                enemy["patrol_target"] = enemy["patrol_start"]
                enemy["patrol_start"] = target
                target = enemy["patrol_target"]

            ex, ey = enemy["pos"]
            tx, ty = target
            
            # Move towards target
            if ex < tx: ex += 1
            elif ex > tx: ex -= 1
            elif ey < ty: ey += 1
            elif ey > ty: ey -= 1
            
            new_enemy_pos = (ex, ey)
            
            # Check for collisions with other enemies
            can_move = True
            for other_enemy in self.enemies:
                if other_enemy is not enemy and other_enemy["pos"] == new_enemy_pos:
                    can_move = False
                    break
            
            if can_move and self.dungeon_map[new_enemy_pos] == 0:
                enemy["pos"] = new_enemy_pos
            
            # Check for collision with player
            if enemy["pos"] == self.player_pos:
                # sfx: player_hurt.wav
                self.player_health -= 10
                self._create_particles(10, self.player_pos, self.COLOR_PLAYER, 1.5, 8, 0.4)
                if self.player_health <= 0:
                    # sfx: player_die.wav
                    self.player_health = 0
                    reward = -100.0
                    terminated = True
                    self.game_over = True
                    break # No more enemy moves if player is dead
        
        # --- Update & Termination Phase ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
        
        # Update timers and particles
        for enemy in self.enemies:
            if enemy["hit_timer"] > 0:
                enemy["hit_timer"] -= 1
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Render dungeon
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.dungeon_map[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # 2. Render exit
        exit_rect = pygame.Rect(self.exit_pos[0] * self.TILE_SIZE, self.exit_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, exit_rect, 1) # Outline

        # 3. Render gold
        for gold in self.gold_piles:
            gx, gy = gold["pos"]
            gold_rect = pygame.Rect(gx * self.TILE_SIZE + self.TILE_SIZE // 4, gy * self.TILE_SIZE + self.TILE_SIZE // 4, self.TILE_SIZE // 2, self.TILE_SIZE // 2)
            pygame.draw.rect(self.screen, self.COLOR_GOLD, gold_rect)
            # Sparkle effect
            if self.np_random.random() < 0.1:
                sx = gx * self.TILE_SIZE + self.np_random.integers(self.TILE_SIZE)
                sy = gy * self.TILE_SIZE + self.np_random.integers(self.TILE_SIZE)
                pygame.gfxdraw.pixel(self.screen, sx, sy, self.COLOR_WHITE)

        # 4. Render enemies
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            bob = math.sin(self.steps * 0.2 + ex) * 2
            enemy_rect = pygame.Rect(ex * self.TILE_SIZE + 2, ey * self.TILE_SIZE + 2 + bob, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
            color = self.COLOR_WHITE if enemy["hit_timer"] > 0 else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, enemy_rect, border_radius=4)
        
        # 5. Render player
        px, py = self.player_pos
        bob = math.sin(self.steps * 0.3) * 2
        player_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE + bob, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        # Eye indicating facing direction
        eye_x = int(player_rect.centerx + self.player_facing[0] * self.TILE_SIZE * 0.25)
        eye_y = int(player_rect.centery + self.player_facing[1] * self.TILE_SIZE * 0.25)
        pygame.draw.circle(self.screen, self.COLOR_WHITE, (eye_x, eye_y), 2)

        # 6. Render particles
        for p in self.particles:
            p_x, p_y = p["pos"]
            pygame.draw.circle(self.screen, p["color"], (int(p_x), int(p_y)), int(p["size"]))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.player_max_health
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold Count
        gold_text = self.font_small.render(f"Gold: {self.gold_collected}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (220, 12))

        # Level
        level_text = self.font_large.render(f"Level: {self.current_level}", True, self.COLOR_UI_TEXT)
        level_rect = level_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(level_text, level_rect)

    def _create_particles(self, count, pos, color, speed, lifetime, spread):
        px, py = pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2, pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_x = math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5)
            vel_y = math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                "pos": [px, py],
                "vel": [vel_x, vel_y],
                "color": color,
                "lifetime": self.np_random.integers(lifetime // 2, lifetime),
                "size": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["size"] = max(0, p["size"] - 0.1)
        self.particles = [p for p in self.particles if p["lifetime"] > 0 and p["size"] > 0]
    
    def _get_info(self):
        return {
            "score": self.gold_collected,
            "steps": self.steps,
            "level": self.current_level,
            "health": self.player_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info

        # Test state guarantees
        self.reset()
        assert self.player_health <= 100
        assert self.gold_collected >= 0
        assert self.current_level <= self.WIN_LEVEL
        assert len(self.enemies) == self.current_level

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Roguelike Dungeon")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Map Pygame keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(env.user_guide)

    while running:
        movement = 0
        space_held = 0
        shift_held = 0 # Unused in this game
        
        # Since this is a turn-based game (auto_advance=False), we wait for an event
        event = pygame.event.wait() 

        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: # Reset game
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")
                continue
            
            if event.key in key_to_action:
                movement = key_to_action[event.key]
            
            if event.key == pygame.K_SPACE:
                space_held = 1
            
            # An action is submitted, so we step the environment
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated or truncated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Optional: auto-reset after a delay
                # pygame.time.wait(2000)
                # obs, info = env.reset()
                # total_reward = 0
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for display, though game logic is turn-based

    env.close()