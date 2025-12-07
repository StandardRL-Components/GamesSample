
# Generated: 2025-08-28T05:36:18.534960
# Source Brief: brief_05630.md
# Brief Index: 5630

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to attack in the direction you last moved."
    )

    game_description = (
        "Escape a procedurally generated dungeon by defeating enemies and reaching the exit within the time limit."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH, self.GRID_HEIGHT = 50, 50

        self.MAX_PLAYER_HEALTH = 5
        self.INITIAL_TIME_LIMIT = 100 # Steps per stage
        self.TOTAL_STAGES = 3

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_FLOOR = (80, 80, 90)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_ENEMY = (50, 150, 255)
        self.COLOR_EXIT = (50, 255, 100)
        self.COLOR_TRAP = (150, 50, 200)
        self.COLOR_TRAP_ACTIVE = (255, 80, 80)
        self.COLOR_ATTACK = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (30, 30, 40, 180)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (150, 50, 50)

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
        try:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 36)
        except pygame.error:
            self.font_small = pygame.font.SysFont("sans", 24)
            self.font_large = pygame.font.SysFont("sans", 36)

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.player_pos = None
        self.player_health = None
        self.player_facing_dir = None
        self.enemies = None
        self.traps = None
        self.exit_pos = None
        self.timer = None
        self.stage = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.attack_animation = None
        self.damage_flash_timer = None

        # --- Persistent State (across resets within an episode) ---
        self.current_stage = 1
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = self.MAX_PLAYER_HEALTH
        self.player_facing_dir = [0, -1] # Start facing up
        self.timer = self.INITIAL_TIME_LIMIT + (self.current_stage - 1) * 20
        
        self._generate_level()

        self.particles = []
        self.attack_animation = {"timer": 0, "pos": (0, 0)}
        self.damage_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1
        self.timer -= 1

        # --- Update Animations ---
        self._update_animations()

        # --- Handle Player Action ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        # 1. Attack
        if space_pressed:
            reward += self._handle_player_attack()
            # SFX: Player attack swing

        # 2. Movement
        moved = self._handle_player_movement(movement)
        
        # --- Update Game World ---
        # Enemies move every other turn
        if self.steps % 2 == 0:
            self._update_enemies()
        
        self._update_traps()

        # --- Check Collisions & Events ---
        # Player-Enemy collision
        for enemy_pos in self.enemies[:]:
            if self.player_pos == enemy_pos:
                self.player_health -= 1
                reward -= 1 # Penalty for collision
                self.enemies.remove(enemy_pos)
                self.damage_flash_timer = 5
                self._create_particles(self.player_pos, self.COLOR_ENEMY)
                # SFX: Player takes damage
                # SFX: Enemy dies
                break
        
        # Player-Trap collision
        for trap_pos, trap_timer in self.traps:
            if self.player_pos == trap_pos and trap_timer == 0:
                self.player_health -= 1
                reward -= 5
                self.damage_flash_timer = 5
                # SFX: Player takes damage from trap
                break

        # --- Check Win/Loss Conditions ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.current_stage = 1 # Reset progress on death
            # SFX: Game over
        
        if self.timer <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.current_stage = 1 # Reset progress on time out
            # SFX: Game over (time out)

        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.game_over = True
            if self.current_stage < self.TOTAL_STAGES:
                self.current_stage += 1
            else:
                self.current_stage = 1 # Loop back after winning
            # SFX: Stage complete

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        # 1. Create a grid of walls
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # 2. Carve out the level with random walks
        num_walks = 10
        walk_length = 200
        for _ in range(num_walks):
            px, py = self.np_random.integers(1, self.GRID_WIDTH-1), self.np_random.integers(1, self.GRID_HEIGHT-1)
            self.grid[px, py] = 0
            dx, dy = self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])
            for _ in range(walk_length):
                if self.np_random.random() < 0.2: # Chance to change direction
                    dx, dy = self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1])
                    if dx == 0 and dy == 0: dx = 1
                
                px, py = np.clip(px + dx, 1, self.GRID_WIDTH - 2), np.clip(py + dy, 1, self.GRID_HEIGHT - 2)
                self.grid[px, py] = 0
                self.grid[px+1, py] = 0 # Make corridors wider
                self.grid[px, py+1] = 0

        # 3. Find all reachable floor tiles using BFS
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        if not floor_tiles: # Failsafe if generation is empty
            self.grid[self.GRID_WIDTH//2, self.GRID_HEIGHT//2] = 0
            floor_tiles = [[self.GRID_WIDTH//2, self.GRID_HEIGHT//2]]

        start_pos = random.choice(floor_tiles)
        q = deque([start_pos])
        reachable = {tuple(start_pos)}
        while q:
            x, y = q.popleft()
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0 and tuple([nx,ny]) not in reachable:
                    reachable.add(tuple([nx,ny]))
                    q.append([nx,ny])
        
        reachable_list = [list(pos) for pos in reachable]
        
        # 4. Place player, exit, enemies, traps
        self.player_pos = random.choice(reachable_list)
        
        # Place exit far from player
        reachable_list.sort(key=lambda p: -math.hypot(p[0]-self.player_pos[0], p[1]-self.player_pos[1]))
        self.exit_pos = reachable_list[0]
        
        # Place enemies
        self.enemies = []
        num_enemies = 2 + self.current_stage * 2
        for _ in range(num_enemies):
            pos = random.choice(reachable_list)
            if pos != self.player_pos and pos != self.exit_pos:
                self.enemies.append(pos)
        
        # Place traps
        self.traps = []
        num_traps = 5 + self.current_stage * 3
        for _ in range(num_traps):
            pos = random.choice(reachable_list)
            if pos != self.player_pos and pos != self.exit_pos and pos not in self.enemies:
                # timer > 0 is countdown to activation, timer == 0 is active
                self.traps.append([pos, self.np_random.integers(10, 30)])

    def _handle_player_movement(self, movement):
        moved = False
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right

        if dx != 0 or dy != 0:
            self.player_facing_dir = [dx, dy]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if self.grid[new_pos[0], new_pos[1]] == 0: # Is floor
                self.player_pos = new_pos
                moved = True
        return moved

    def _handle_player_attack(self):
        attack_pos = [self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1]]
        self.attack_animation = {"timer": 3, "pos": attack_pos}
        reward = 0
        for enemy_pos in self.enemies[:]:
            if enemy_pos == attack_pos:
                self.enemies.remove(enemy_pos)
                reward += 1
                self._create_particles(enemy_pos, self.COLOR_ENEMY)
                # SFX: Enemy hit/death
        return reward

    def _update_enemies(self):
        for i, enemy_pos in enumerate(self.enemies):
            dx = self.player_pos[0] - enemy_pos[0]
            dy = self.player_pos[1] - enemy_pos[1]
            
            move_x, move_y = 0, 0
            if abs(dx) > abs(dy):
                move_x = np.sign(dx)
            else:
                move_y = np.sign(dy)
            
            if move_x == 0 and move_y == 0: continue # On same tile
            
            new_pos = [enemy_pos[0] + move_x, enemy_pos[1] + move_y]
            if self.grid[new_pos[0], new_pos[1]] == 0:
                is_occupied = False
                for other_enemy in self.enemies:
                    if other_enemy == new_pos:
                        is_occupied = True
                        break
                if not is_occupied:
                    self.enemies[i] = new_pos

    def _update_traps(self):
        for i in range(len(self.traps)):
            pos, timer = self.traps[i]
            if timer > 0:
                self.traps[i][1] -= 1
            else: # timer is 0, trap is active. Reset it after a short duration.
                if self.np_random.random() < 0.1:
                    self.traps[i][1] = self.np_random.integers(20, 50) # Reset timer

    def _update_animations(self):
        # Attack
        if self.attack_animation["timer"] > 0:
            self.attack_animation["timer"] -= 1
        # Damage flash
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1
        # Particles
        for p in self.particles[:]:
            p["timer"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            if p["timer"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(10):
            self.particles.append({
                "pos": [px + self.TILE_SIZE/2, py + self.TILE_SIZE/2],
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)],
                "timer": self.np_random.integers(10, 20),
                "color": color
            })

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
            "health": self.player_health,
            "stage": self.current_stage,
            "time_left": self.timer,
        }

    def _grid_to_pixel(self, grid_pos):
        return grid_pos[0] * self.TILE_SIZE, grid_pos[1] * self.TILE_SIZE
    
    def _render_game(self):
        # Camera offset to center player
        cam_offset_x = self.WIDTH // 2 - (self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2)
        cam_offset_y = self.HEIGHT // 2 - (self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2)

        # Visible grid range
        start_gx = max(0, (self.player_pos[0] - self.WIDTH // (2*self.TILE_SIZE)) - 5)
        end_gx = min(self.GRID_WIDTH, (self.player_pos[0] + self.WIDTH // (2*self.TILE_SIZE)) + 5)
        start_gy = max(0, (self.player_pos[1] - self.HEIGHT // (2*self.TILE_SIZE)) - 5)
        end_gy = min(self.GRID_HEIGHT, (self.player_pos[1] + self.HEIGHT // (2*self.TILE_SIZE)) + 5)

        # Render dungeon
        for x in range(start_gx, end_gx):
            for y in range(start_gy, end_gy):
                px, py = x * self.TILE_SIZE + cam_offset_x, y * self.TILE_SIZE + cam_offset_y
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (px, py, self.TILE_SIZE, self.TILE_SIZE))

        # Render traps
        for pos, timer in self.traps:
            px, py = pos[0] * self.TILE_SIZE + cam_offset_x, pos[1] * self.TILE_SIZE + cam_offset_y
            if 0 < timer <= 5 and self.steps % 2 == 0: # Warning flash
                color = self.COLOR_TRAP_ACTIVE
            elif timer == 0: # Active
                color = self.COLOR_TRAP_ACTIVE
            else:
                color = self.COLOR_TRAP
            pygame.draw.rect(self.screen, color, (px + 4, py + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))
        
        # Render exit
        ex, ey = self.exit_pos[0] * self.TILE_SIZE + cam_offset_x, self.exit_pos[1] * self.TILE_SIZE + cam_offset_y
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, self.TILE_SIZE, self.TILE_SIZE))
        pygame.gfxdraw.rectangle(self.screen, (ex, ey, self.TILE_SIZE, self.TILE_SIZE), (255,255,255))


        # Render enemies
        for pos in self.enemies:
            px, py = pos[0] * self.TILE_SIZE + cam_offset_x, pos[1] * self.TILE_SIZE + cam_offset_y
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))

        # Render player
        px, py = self.player_pos[0] * self.TILE_SIZE + cam_offset_x, self.player_pos[1] * self.TILE_SIZE + cam_offset_y
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        pygame.gfxdraw.rectangle(self.screen, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4), (255,255,255))

        # Render attack animation
        if self.attack_animation["timer"] > 0:
            pos = self.attack_animation["pos"]
            px, py = pos[0] * self.TILE_SIZE + cam_offset_x, pos[1] * self.TILE_SIZE + cam_offset_y
            alpha = (self.attack_animation["timer"] / 3) * 255
            s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            s.fill((*self.COLOR_ATTACK, alpha))
            self.screen.blit(s, (px, py))
            
        # Render particles
        for p in self.particles:
            size = (p["timer"] / 20) * 5
            pygame.draw.rect(self.screen, p["color"], (p["pos"][0] + cam_offset_x, p["pos"][1] + cam_offset_y, size, size))

        # Render damage flash
        if self.damage_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = (self.damage_flash_timer / 5) * 100
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # UI Background
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0,0))
        
        # Health Bar
        health_ratio = self.player_health / self.MAX_PLAYER_HEALTH
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.MAX_PLAYER_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Timer
        timer_text = self.font_large.render(f"TIME: {self.timer}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))

        # Stage
        stage_text = self.font_large.render(f"STAGE: {self.current_stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 5))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            msg = "STAGE CLEAR" if self.player_health > 0 and self.timer > 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))


    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dungeon Escape")
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
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Resetting environment...")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(10) # Control the speed of the manual play

    env.close()