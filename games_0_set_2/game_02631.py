
# Generated: 2025-08-28T05:28:20.001728
# Source Brief: brief_02631.md
# Brief Index: 2631

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the worm. Avoid the red enemies and reach the green exit!"
    )

    game_description = (
        "Guide a worm through a procedurally generated cave, dodging enemies and reaching the exit as quickly as possible."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.MAX_STAGES = 3
        self.WORM_SPEED = 2
        self.WORM_LENGTH = 15
        
        # Colors
        self.COLOR_BG = (15, 10, 5)
        self.COLOR_WALL = (60, 40, 20)
        self.COLOR_WORM_HEAD = (100, 255, 100)
        self.COLOR_WORM_BODY = (50, 200, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_EXIT = (150, 255, 150)
        self.COLOR_PARTICLE = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.worm_segments = None
        self.enemies = None
        self.wall_rects = None
        self.exit_rect = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.enemy_speed = 1.0
        self.game_over = False
        self.np_random = None

        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        if options and options.get("soft_reset", False):
            # This is for advancing to the next stage
            pass
        else:
            # Full reset for a new episode
            self.score = 0
            self.stage = 1
            self.enemy_speed = 1.0

        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Create a grid for cellular automata
        grid_w, grid_h = self.WIDTH // 10, self.HEIGHT // 10
        grid = self.np_random.choice([0, 1], size=(grid_w, grid_h), p=[0.55, 0.45])

        # Cellular automata to smooth out the caves
        for _ in range(4):
            new_grid = grid.copy()
            for y in range(1, grid_h - 1):
                for x in range(1, grid_w - 1):
                    neighbors = np.sum(grid[x-1:x+2, y-1:y+2]) - grid[x, y]
                    if grid[x, y] == 1 and neighbors < 3:
                        new_grid[x, y] = 0
                    elif grid[x, y] == 0 and neighbors > 4:
                        new_grid[x, y] = 1
            grid = new_grid

        # Ensure borders are walls
        grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1] = 1, 1, 1, 1

        # Find the largest open area using flood fill
        max_area = 0
        best_fill = None
        visited = np.zeros_like(grid, dtype=bool)
        for y in range(grid_h):
            for x in range(grid_w):
                if grid[x, y] == 0 and not visited[x, y]:
                    area = 0
                    q = deque([(x, y)])
                    fill = []
                    current_visited = set([(x,y)])
                    
                    while q:
                        cx, cy = q.popleft()
                        if visited[cx,cy]: continue
                        
                        visited[cx, cy] = True
                        area += 1
                        fill.append((cx, cy))
                        
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < grid_w and 0 <= ny < grid_h and \
                               grid[nx, ny] == 0 and not visited[nx, ny] and (nx,ny) not in current_visited:
                                q.append((nx, ny))
                                current_visited.add((nx,ny))
                    
                    if area > max_area:
                        max_area = area
                        best_fill = fill

        # Finalize cave structure
        final_grid = np.ones_like(grid)
        open_coords = []
        if best_fill:
            for x, y in best_fill:
                final_grid[x, y] = 0
                open_coords.append((x * 10, y * 10))

        # Create wall rects from the final grid
        self.wall_rects = []
        for y in range(grid_h):
            for x in range(grid_w):
                if final_grid[x, y] == 1:
                    self.wall_rects.append(pygame.Rect(x * 10, y * 10, 10, 10))

        # Place start and exit
        start_pos_grid = min(open_coords, key=lambda p: p[0])
        self.exit_pos_grid = max(open_coords, key=lambda p: p[0])
        self.exit_rect = pygame.Rect(self.exit_pos_grid[0], self.exit_pos_grid[1], 10, 10)

        # Place worm
        self.worm_segments = deque([pygame.Rect(start_pos_grid[0] - i * self.WORM_SPEED, start_pos_grid[1], 8, 8) for i in range(self.WORM_LENGTH)])

        # Place enemies
        self.enemies = []
        for _ in range(5):
            for _ in range(100): # Max attempts to find a spot
                pos = random.choice(open_coords)
                # Ensure enemies are not too close to start or exit
                if math.hypot(pos[0] - start_pos_grid[0], pos[1] - start_pos_grid[1]) > 100 and \
                   math.hypot(pos[0] - self.exit_pos_grid[0], pos[1] - self.exit_pos_grid[1]) > 100:
                    
                    patrol_radius = self.np_random.integers(20, 50)
                    enemy = {
                        "rect": pygame.Rect(pos[0], pos[1], 10, 10),
                        "angle": self.np_random.uniform(0, 2 * math.pi),
                        "center": pos,
                        "radius": patrol_radius
                    }
                    self.enemies.append(enemy)
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        self.steps += 1
        reward = -0.01  # Time penalty

        # --- Game Logic ---
        old_head_pos = self.worm_segments[0].center
        
        # 1. Move Worm
        self._move_worm(movement)

        # 2. Move Enemies
        self._move_enemies()

        # 3. Update Particles
        self._update_particles()
        
        # 4. Calculate distance-based reward
        new_head_pos = self.worm_segments[0].center
        prev_dist = math.hypot(old_head_pos[0] - self.exit_rect.centerx, old_head_pos[1] - self.exit_rect.centery)
        new_dist = math.hypot(new_head_pos[0] - self.exit_rect.centerx, new_head_pos[1] - self.exit_rect.centery)
        if new_dist < prev_dist:
            reward += 0.1

        # 5. Check for collisions and win/loss conditions
        terminated = False
        
        # Check wall collision
        if self.worm_segments[0].collidelist(self.wall_rects) != -1:
            self._create_explosion(self.worm_segments[0].center)
            reward -= 10
            self.game_over = True
            terminated = True
        
        # Check enemy collision
        for enemy in self.enemies:
            if self.worm_segments[0].colliderect(enemy["rect"]):
                self._create_explosion(self.worm_segments[0].center)
                reward -= 10
                self.game_over = True
                terminated = True
                break
        
        # Check win condition
        if not terminated and self.worm_segments[0].colliderect(self.exit_rect):
            win_bonus = 10 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += 10 + win_bonus
            self.score += int(10 + win_bonus)
            
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self.enemy_speed += 0.2
                self.reset(options={"soft_reset": True})
            else:
                self.game_over = True
                terminated = True

        # Check step limit
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_worm(self, movement):
        head = self.worm_segments[0].copy()
        dx, dy = 0, 0
        if movement == 1: dy = -self.WORM_SPEED
        elif movement == 2: dy = self.WORM_SPEED
        elif movement == 3: dx = -self.WORM_SPEED
        elif movement == 4: dx = self.WORM_SPEED
        
        if dx != 0 or dy != 0:
            head.move_ip(dx, dy)
            # Clamp to screen bounds
            head.left = max(0, min(head.left, self.WIDTH - head.width))
            head.top = max(0, min(head.top, self.HEIGHT - head.height))
            self.worm_segments.appendleft(head)
            self.worm_segments.pop()

    def _move_enemies(self):
        for enemy in self.enemies:
            enemy["angle"] += 0.05
            target_x = enemy["center"][0] + math.cos(enemy["angle"]) * enemy["radius"]
            target_y = enemy["center"][1] + math.sin(enemy["angle"]) * enemy["radius"]
            
            dx = target_x - enemy["rect"].centerx
            dy = target_y - enemy["rect"].centery
            dist = math.hypot(dx, dy)
            
            if dist > 0:
                enemy["rect"].x += dx / dist * self.enemy_speed
                enemy["rect"].y += dy / dist * self.enemy_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_explosion(self, pos):
        # Sound effect placeholder: # sfx.play('explosion')
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'size': self.np_random.integers(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Walls
        for wall in self.wall_rects:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render Exit (with glow)
        glow_size = int(15 + 5 * math.sin(self.steps * 0.1))
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_EXIT, 50), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (self.exit_rect.centerx - glow_size, self.exit_rect.centery - glow_size))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Render Enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy["rect"])
            pygame.gfxdraw.rectangle(self.screen, enemy["rect"], (*self.COLOR_ENEMY, 150))

        # Render Worm
        for i, segment in enumerate(self.worm_segments):
            color = self.COLOR_WORM_HEAD if i == 0 else self.COLOR_WORM_BODY
            pygame.draw.rect(self.screen, color, segment, border_radius=3)
        
        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            pygame.draw.circle(self.screen, (*self.COLOR_PARTICLE, alpha), [int(c) for c in p['pos']], p['size'])

    def _render_ui(self):
        stage_text = self.font_small.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        timer_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
        }

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
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows'

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cave Worm")
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
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space and Shift are not used in this game, but we pass them
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()