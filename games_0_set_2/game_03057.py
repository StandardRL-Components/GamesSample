
# Generated: 2025-08-28T06:51:56.883746
# Source Brief: brief_03057.md
# Brief Index: 3057

        
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

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your square. Collect red fruits while avoiding orange triangles."
    )
    game_description = (
        "A fast-paced arcade game. Navigate a grid to collect fruits for points. "
        "Dodge moving obstacles, as collision ends the game. "
        "Higher scores are awarded for collecting fruits near obstacles."
    )

    # Frame advance setting
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.CELL_SIZE = 20
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255)
        self.COLOR_FRUIT = (255, 60, 60)
        self.COLOR_OBSTACLE = (255, 165, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_PARTICLE = (255, 255, 100)

        # Game settings
        self.INITIAL_FRUIT_COUNT = 5
        self.INITIAL_OBSTACLE_COUNT = 4
        self.WIN_CONDITION_FRUITS = 50
        self.MAX_STEPS = 1000
        self.INITIAL_OBSTACLE_SPEED = 0.5
        self.OBSTACLE_SPEED_INCREMENT = 0.05

        # Reward values
        self.REWARD_COLLECT_FRUIT = 10.0
        self.REWARD_RISKY_COLLECT_BONUS = 10.0
        self.REWARD_WIN = 50.0
        self.REWARD_COLLISION = -50.0
        self.REWARD_CLOSER_TO_FRUIT = 1.0
        self.REWARD_CLOSER_TO_OBSTACLE = -0.1
        self.REWARD_SAFE_MOVE = -2.0

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_info = pygame.font.SysFont("monospace", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_info = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.fruits = []
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = 0.0
        self.steps = 0
        self.score = 0
        self.collected_fruit_count = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.collected_fruit_count = 0
        self.game_over = False
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.particles = []

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.fruits = []
        for _ in range(self.INITIAL_FRUIT_COUNT):
            self._spawn_fruit()

        self.obstacles = []
        for _ in range(self.INITIAL_OBSTACLE_COUNT):
            self._spawn_obstacle()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        last_player_pos = self.player_pos.copy()

        self._update_player_position(movement)
        self._update_obstacles()
        self._update_particles()
        
        reward, terminated = self._resolve_game_events(last_player_pos)
        self.score += reward
        self.game_over = terminated

        # Final check for max steps termination
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player_position(self, movement):
        if movement == 1: self.player_pos[1] -= 1  # Up
        elif movement == 2: self.player_pos[1] += 1  # Down
        elif movement == 3: self.player_pos[0] -= 1  # Left
        elif movement == 4: self.player_pos[0] += 1  # Right
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'][0] += obs['vel'][0] * self.obstacle_speed
            obs['pos'][1] += obs['vel'][1] * self.obstacle_speed

            if obs['pos'][0] < 0 or obs['pos'][0] >= self.GRID_WIDTH - 1:
                obs['vel'][0] *= -1
                obs['pos'][0] = np.clip(obs['pos'][0], 0, self.GRID_WIDTH - 1)
            if obs['pos'][1] < 0 or obs['pos'][1] >= self.GRID_HEIGHT - 1:
                obs['vel'][1] *= -1
                obs['pos'][1] = np.clip(obs['pos'][1], 0, self.GRID_HEIGHT - 1)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _resolve_game_events(self, last_pos):
        if self._check_player_obstacle_collision():
            return self.REWARD_COLLISION, True

        collected_fruit_index = self._get_collected_fruit_index()
        if collected_fruit_index is not None:
            collected_fruit_pos = self.fruits.pop(collected_fruit_index)
            self.collected_fruit_count += 1
            
            # Sound effect placeholder
            # pygame.mixer.Sound.play(collect_sound)
            self._spawn_particles(self._grid_to_pixel(collected_fruit_pos, center=True))
            self._spawn_fruit()

            if self.collected_fruit_count > 0 and self.collected_fruit_count % 10 == 0:
                self.obstacle_speed += self.OBSTACLE_SPEED_INCREMENT

            reward = self.REWARD_COLLECT_FRUIT
            if self._is_risky_collection(collected_fruit_pos):
                reward += self.REWARD_RISKY_COLLECT_BONUS

            if self.collected_fruit_count >= self.WIN_CONDITION_FRUITS:
                reward += self.REWARD_WIN
                return reward, True
            return reward, False
            
        reward = self.REWARD_SAFE_MOVE
        if self.fruits:
            dist_fruit_before = min(self._manhattan_distance(last_pos, f) for f in self.fruits)
            dist_fruit_after = min(self._manhattan_distance(self.player_pos, f) for f in self.fruits)
            if dist_fruit_after < dist_fruit_before:
                reward += self.REWARD_CLOSER_TO_FRUIT

        if self.obstacles:
            dist_obst_before = min(self._manhattan_distance(last_pos, [int(o['pos'][0]), int(o['pos'][1])]) for o in self.obstacles)
            dist_obst_after = min(self._manhattan_distance(self.player_pos, [int(o['pos'][0]), int(o['pos'][1])]) for o in self.obstacles)
            if dist_obst_after < dist_obst_before:
                reward += self.REWARD_CLOSER_TO_OBSTACLE
                
        return reward, False

    def _check_player_obstacle_collision(self):
        for obs in self.obstacles:
            if self.player_pos == [int(obs['pos'][0]), int(obs['pos'][1])]:
                # Sound effect placeholder
                # pygame.mixer.Sound.play(collision_sound)
                return True
        return False

    def _get_collected_fruit_index(self):
        for i, fruit_pos in enumerate(self.fruits):
            if self.player_pos == fruit_pos:
                return i
        return None

    def _is_risky_collection(self, fruit_pos):
        for obs in self.obstacles:
            if self._manhattan_distance(fruit_pos, [int(obs['pos'][0]), int(obs['pos'][1])]) <= 1:
                return True
        return False

    def _spawn_fruit(self):
        pos = None
        while pos is None:
            new_pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            
            occupied_by_player = (new_pos == self.player_pos)
            occupied_by_fruit = any(new_pos == f for f in self.fruits)
            occupied_by_obstacle = any(new_pos == [int(o['pos'][0]), int(o['pos'][1])] for o in self.obstacles)

            if not (occupied_by_player or occupied_by_fruit or occupied_by_obstacle):
                pos = new_pos
        self.fruits.append(pos)
        
    def _spawn_obstacle(self):
        pos = None
        while pos is None:
            new_pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if self._manhattan_distance(new_pos, self.player_pos) > 3:
                pos = new_pos
        
        vel = self.np_random.choice([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.obstacles.append({'pos': [float(pos[0]), float(pos[1])], 'vel': vel.tolist()})

    def _spawn_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(10, 20)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw fruits
        for fruit_pos in self.fruits:
            px, py = self._grid_to_pixel(fruit_pos, center=True)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(self.CELL_SIZE * 0.4), self.COLOR_FRUIT)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(self.CELL_SIZE * 0.4), self.COLOR_FRUIT)

        # Draw obstacles
        for obs in self.obstacles:
            px, py = self._grid_to_pixel(obs['pos'])
            s = self.CELL_SIZE
            p1 = (int(px + s/2), int(py + s*0.1))
            p2 = (int(px + s*0.1), int(py + s*0.9))
            p3 = (int(px + s*0.9), int(py + s*0.9))
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_OBSTACLE)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_OBSTACLE)
            
        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        s = self.CELL_SIZE
        glow_size = int(s * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 60), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (player_px + s/2 - glow_size//2, player_py + s/2 - glow_size//2))
        
        player_rect = pygame.Rect(player_px, player_py, s, s)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-s*0.2, -s*0.2), border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*self.COLOR_PARTICLE[:3], alpha))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        fruit_text = self.font_info.render(f"FRUIT: {self.collected_fruit_count} / {self.WIN_CONDITION_FRUITS}", True, self.COLOR_TEXT)
        self.screen.blit(fruit_text, (20, 50))
        
        steps_text = self.font_info.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.collected_fruit_count >= self.WIN_CONDITION_FRUITS else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_fruits": self.collected_fruit_count
        }

    def _grid_to_pixel(self, grid_pos, center=False):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE / 2
            py += self.CELL_SIZE / 2
        return px, py

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.quit()
        
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        # Player input
        movement = 0 # No-op by default
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            # Since auto_advance is False, we only step on an action.
            # For human play, we can step every frame if a key is held, or on key presses.
            # Here, we step on every frame to feel responsive.
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control human play speed

    env.close()
    pygame.quit()