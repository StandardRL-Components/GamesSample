
# Generated: 2025-08-28T04:02:51.908606
# Source Brief: brief_02199.md
# Brief Index: 2199

        
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
        "Controls: Arrow keys to move. Hold space to attack in the direction you are facing. "
        "Find the glowing green exit before the timer runs out."
    )

    game_description = (
        "Navigate a procedurally generated crypt, battling monsters and seeking the exit before time runs out. "
        "Collect yellow health packs to survive."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = 20
        self.MAX_TURNS = 60
        self.MAX_EPISODE_STEPS = 1800 # 3 stages * 60 turns/stage * 10 steps/turn (approx) - not used for termination
        self.MAX_STAGES = 3

        # Colors
        self.COLOR_BG = (20, 15, 25)
        self.COLOR_WALL = (40, 30, 50)
        self.COLOR_FLOOR = (60, 50, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (220, 40, 40)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_RESOURCE = (255, 220, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        self.COLOR_HEALTH_BAR = (40, 200, 40)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_HIT_FLASH = (255, 255, 255)

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("unifont", 18)
            self.font_large = pygame.font.SysFont("unifont", 24)
        except pygame.error:
            self.font_small = pygame.font.SysFont("monospace", 18)
            self.font_large = pygame.font.SysFont("monospace", 24)

        # Game state variables (initialized in reset)
        self.grid = None
        self.player_pos = None
        self.player_health = None
        self.player_max_health = 10
        self.player_facing = None
        self.player_hit_timer = 0
        self.enemies = []
        self.resources = []
        self.exit_pos = None
        self.turn_timer = None
        self.stage = None
        self.enemy_spawn_rate = None
        self.turns_since_spawn = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        # 1. Fill with walls
        self.grid = np.ones((self.GRID_W, self.GRID_H)) # 1 for wall, 0 for floor

        # 2. Carve rooms
        rooms = []
        for _ in range(random.randint(5, 8)):
            w, h = random.randint(3, 7), random.randint(3, 7)
            x, y = random.randint(1, self.GRID_W - w - 1), random.randint(1, self.GRID_H - h - 1)
            self.grid[x:x+w, y:y+h] = 0
            rooms.append(pygame.Rect(x, y, w, h))

        # 3. Connect rooms
        for i in range(len(rooms) - 1):
            x1, y1 = rooms[i].center
            x2, y2 = rooms[i+1].center
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.grid[x, y1] = 0
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.grid[x2, y] = 0

        floor_tiles = list(zip(*np.where(self.grid == 0)))
        random.shuffle(floor_tiles)

        # 4. Place player
        self.player_pos = floor_tiles.pop()

        # 5. Place exit
        best_pos, max_dist = None, -1
        for pos in floor_tiles:
            dist = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
            if dist > max_dist:
                max_dist = dist
                best_pos = pos
        self.exit_pos = best_pos
        floor_tiles.remove(best_pos)
        
        # 6. Place resources
        self.resources = []
        for _ in range(random.randint(2, 4)):
            if floor_tiles:
                self.resources.append(floor_tiles.pop())

        # 7. Place initial enemies
        self.enemies = []
        for _ in range(self.stage): # More enemies on later stages
            if floor_tiles:
                pos = floor_tiles.pop()
                self.enemies.append({"pos": pos, "health": 2, "hit_timer": 0})
    
    def _next_stage(self):
        self.stage += 1
        self.turn_timer = self.MAX_TURNS
        self.enemy_spawn_rate = max(2, 5 - (self.stage - 1))
        self.turns_since_spawn = 0
        self.player_facing = (0, -1)
        self.particles = []
        self._generate_level()
        # Health persists, but capped
        self.player_health = min(self.player_max_health, self.player_health)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.stage = 1
        self.player_health = self.player_max_health
        
        self.turn_timer = self.MAX_TURNS
        self.enemy_spawn_rate = max(2, 5 - (self.stage - 1))
        self.turns_since_spawn = 0
        self.player_facing = (0, -1)
        self.particles = []
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward per turn
        self.steps += 1
        
        # --- 1. Player Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Update facing direction from movement input
        if movement == 1: self.player_facing = (0, -1) # Up
        elif movement == 2: self.player_facing = (0, 1)  # Down
        elif movement == 3: self.player_facing = (-1, 0) # Left
        elif movement == 4: self.player_facing = (1, 0)  # Right

        # Attack takes precedence
        if space_held:
            # sfx: sword_swing.wav
            attack_pos = (self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1])
            self._create_particles(attack_pos, self.COLOR_ATTACK, 10, 0.2)
            
            for enemy in self.enemies[:]:
                if enemy["pos"] == attack_pos:
                    enemy["health"] -= 1
                    enemy["hit_timer"] = 3 # frames for flash
                    if enemy["health"] <= 0:
                        # sfx: enemy_die.wav
                        reward += 1.0
                        self.score += 10
                        self._create_particles(enemy["pos"], self.COLOR_ENEMY, 30, 0.5)
                        self.enemies.remove(enemy)
        # Then movement
        elif movement > 0:
            next_pos = (self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1])
            if self.grid[next_pos[0], next_pos[1]] == 0: # Is a floor tile
                self.player_pos = next_pos
                # sfx: step.wav
            else:
                # sfx: bump_wall.wav
                pass
        
        # --- 2. Enemy Turn ---
        for enemy in self.enemies:
            if enemy["hit_timer"] > 0: enemy["hit_timer"] -=1
            
            px, py = self.player_pos
            ex, ey = enemy["pos"]
            
            # Attack if adjacent
            if abs(px - ex) + abs(py - ey) == 1:
                # sfx: player_hurt.wav
                self.player_health -= 1
                self.player_hit_timer = 3 # frames for flash
                reward -= 0.2
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 15, 0.3, life=10)
            # Move towards player
            else:
                dx, dy = np.sign(px - ex), np.sign(py - ey)
                # Simple greedy move, alternating axes to avoid getting stuck
                if self.steps % 2 == 0: # Move X first
                    if dx != 0 and self.grid[ex + dx, ey] == 0:
                        enemy["pos"] = (ex + dx, ey)
                    elif dy != 0 and self.grid[ex, ey + dy] == 0:
                        enemy["pos"] = (ex, ey + dy)
                else: # Move Y first
                    if dy != 0 and self.grid[ex, ey + dy] == 0:
                        enemy["pos"] = (ex, ey + dy)
                    elif dx != 0 and self.grid[ex + dx, ey] == 0:
                        enemy["pos"] = (ex + dx, ey)

        if self.player_hit_timer > 0: self.player_hit_timer -= 1

        # --- 3. State & Interaction Updates ---
        self.turn_timer -= 1
        self.turns_since_spawn += 1

        # Check resource pickup
        if self.player_pos in self.resources:
            # sfx: pickup_health.wav
            self.resources.remove(self.player_pos)
            self.player_health = min(self.player_max_health, self.player_health + 2)
            reward += 5.0
            self.score += 25
            self._create_particles(self.player_pos, self.COLOR_RESOURCE, 20, 0.4)

        # Check enemy spawn
        if self.turns_since_spawn >= self.enemy_spawn_rate:
            floor_tiles = list(zip(*np.where(self.grid == 0)))
            # Don't spawn on player or other entities
            occupied = [self.player_pos] + [e["pos"] for e in self.enemies] + self.resources
            valid_spawns = [p for p in floor_tiles if p not in occupied and (abs(p[0]-self.player_pos[0]) + abs(p[1]-self.player_pos[1])) > 5]
            if valid_spawns:
                pos = random.choice(valid_spawns)
                self.enemies.append({"pos": pos, "health": 2, "hit_timer": 0})
                self.turns_since_spawn = 0

        # --- 4. Termination & Stage Clear ---
        terminated = False
        # Check exit
        if self.player_pos == self.exit_pos:
            if self.stage >= self.MAX_STAGES:
                # sfx: game_win.wav
                reward += 100.0
                self.score += 1000
                terminated = True
                self.game_over = True
            else:
                # sfx: stage_clear.wav
                self.score += 100
                self._next_stage()

        if self.player_health <= 0 or self.turn_timer <= 0:
            if self.player_health > 0: # If lost by timeout, not death
                reward -= 100.0
            else: # Lost by death
                reward -= 100.0
            # sfx: game_over.wav
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, color, count, speed, life=20):
        px, py = (pos[0] + 0.5) * self.TILE_SIZE, (pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.1, speed)
            vel = (math.cos(angle) * s, math.sin(angle) * s)
            self.particles.append([
                [px, py], vel, random.randint(2, 5), color, life
            ])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0] * self.TILE_SIZE * 0.1
            p[0][1] += p[1][1] * self.TILE_SIZE * 0.1
            p[4] -= 1 # Decrement life
            if p[4] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit with glow
        ex, ey = self.exit_pos
        center = (int((ex + 0.5) * self.TILE_SIZE), int((ey + 0.5) * self.TILE_SIZE))
        for i in range(15, 0, -2):
            alpha = 100 - i * 5
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(i * 0.8), color)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(self.TILE_SIZE * 0.4), self.COLOR_EXIT)

        # Draw resources
        for rx, ry in self.resources:
            center = (int((rx + 0.5) * self.TILE_SIZE), int((ry + 0.5) * self.TILE_SIZE))
            angle = (self.steps % 30 / 30) * math.pi * 2
            for i in range(4):
                a = angle + i * math.pi / 2
                start = (center[0] + math.cos(a) * 4, center[1] + math.sin(a) * 4)
                end = (center[0] + math.cos(a) * 8, center[1] + math.sin(a) * 8)
                pygame.draw.line(self.screen, self.COLOR_RESOURCE, start, end, 2)
        
        # Draw particles
        self._update_particles()
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 20))))
            color = (*p[3], alpha)
            surf = pygame.Surface((p[2]*2, p[2]*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p[2], p[2]), p[2])
            self.screen.blit(surf, (int(p[0][0]-p[2]), int(p[0][1]-p[2])))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            rect = (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            color = self.COLOR_HIT_FLASH if enemy["hit_timer"] > 0 else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, rect)

        # Draw player
        px, py = self.player_pos
        rect = (px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        color = self.COLOR_HIT_FLASH if self.player_hit_timer > 0 else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, color, rect)
        
        # Draw player facing indicator
        fx, fy = self.player_facing
        center_x, center_y = (px + 0.5) * self.TILE_SIZE, (py + 0.5) * self.TILE_SIZE
        indicator_pos = (center_x + fx * 8, center_y + fy * 8)
        pygame.draw.circle(self.screen, self.COLOR_TEXT, (int(indicator_pos[0]), int(indicator_pos[1])), 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.player_max_health)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Timer
        timer_text = self.font_large.render(f"Turns: {self.turn_timer}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Stage & Score
        stage_text = self.font_small.render(f"Crypt Level: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, self.HEIGHT - stage_text.get_height() - 10))
        
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, self.HEIGHT - score_text.get_height() - 10))

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
            "stage": self.stage,
            "health": self.player_health,
            "timer": self.turn_timer,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game with keyboard controls
    # when the script is run directly.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crypt Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    total_reward = 0
    total_steps = 0
    
    print(env.user_guide)
    print(env.game_description)

    # Use a dictionary to track keydown events to register one action per press
    key_action_map = {
        pygame.K_UP: (1, 0, 0),
        pygame.K_DOWN: (2, 0, 0),
        pygame.K_LEFT: (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
    }

    while running:
        action_to_take = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_action_map:
                    move, space, shift = key_action_map[event.key]
                    # Get current facing direction for spacebar attack
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: move = 1
                    elif keys[pygame.K_DOWN]: move = 2
                    elif keys[pygame.K_LEFT]: move = 3
                    elif keys[pygame.K_RIGHT]: move = 4
                    action_to_take = [move, space, shift]
                elif event.key == pygame.K_w: # Wait action
                    action_to_take = [0, 0, 0]

        if action_to_take is not None:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            total_steps += 1
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {total_steps}")
                obs, info = env.reset()
                total_reward = 0
                total_steps = 0
                pygame.time.wait(2000)

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)

    env.close()