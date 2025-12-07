import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:51:40.358030
# Source Brief: brief_02427.md
# Brief Index: 2427
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A two-player, turn-based snake game where players compete to collect the most apples.
    
    The environment features:
    - A red "fast" snake and a blue "slow" snake, with alternating turns.
    - Procedurally generated apples that respawn upon collection.
    - A strategic element where snakes cannot move into walls or other snake segments.
    - High-quality visuals with smooth rendering, particle effects, and glowing objects.
    - A clear UI displaying scores and the current turn.
    - A reward structure designed for reinforcement learning.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A two-player, turn-based snake game where players compete to collect the most apples."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to change your snake's direction on your turn."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 500
        self.NUM_APPLES = 5
        
        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_RED = (255, 70, 70)
        self.COLOR_BLUE = (70, 150, 255)
        self.COLOR_APPLE = (80, 255, 80)
        self.COLOR_APPLE_GLOW = (80, 255, 80)
        self.COLOR_TEXT = (220, 220, 220)
        
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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.red_score = None
        self.blue_score = None
        self.game_over = None
        self.current_player = None
        
        self.red_snake_body = None
        self.blue_snake_body = None
        self.red_snake_dir = None
        self.blue_snake_dir = None
        
        self.apples = None
        self.particles = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.red_score = 0
        self.blue_score = 0
        self.game_over = False
        self.current_player = 'red' # Red snake starts
        
        # Initialize snakes
        self.red_snake_body = [(self.GRID_W // 4, self.GRID_H // 2)]
        self.blue_snake_body = [(self.GRID_W * 3 // 4, self.GRID_H // 2)]
        self.red_snake_dir = (1, 0)  # Moving right
        self.blue_snake_dir = (-1, 0) # Moving left
        
        # Initialize apples
        self.apples = []
        for _ in range(self.NUM_APPLES):
            self._spawn_apple()
            
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        # --- Determine Active Player ---
        if self.current_player == 'red':
            active_snake_body = self.red_snake_body
            active_snake_dir = self.red_snake_dir
            speed = 2
        else: # 'blue'
            active_snake_body = self.blue_snake_body
            active_snake_dir = self.blue_snake_dir
            speed = 1
            
        # --- Calculate Movement ---
        new_dir = active_snake_dir
        moved = False
        if movement == 1 and active_snake_dir != (0, 1): new_dir = (0, -1); moved=True # Up
        elif movement == 2 and active_snake_dir != (0, -1): new_dir = (0, 1); moved=True  # Down
        elif movement == 3 and active_snake_dir != (1, 0): new_dir = (-1, 0); moved=True # Left
        elif movement == 4 and active_snake_dir != (-1, 0): new_dir = (1, 0); moved=True  # Right
        
        # Red snake cannot reverse
        if self.current_player == 'red':
            if movement == 1 and active_snake_dir == (0, 1): moved=False
            if movement == 2 and active_snake_dir == (0, -1): moved=False
            if movement == 3 and active_snake_dir == (1, 0): moved=False
            if movement == 4 and active_snake_dir == (-1, 0): moved=False
        
        old_head = active_snake_body[0]
        new_head = old_head
        
        # The fast snake moves two steps if a direction is chosen
        if moved:
            if self.current_player == 'red':
                temp_head = (old_head[0] + new_dir[0], old_head[1] + new_dir[1])
                new_head = (temp_head[0] + new_dir[0], temp_head[1] + new_dir[1])
            else:
                 new_head = (old_head[0] + new_dir[0], old_head[1] + new_dir[1])

        # --- Collision Detection ---
        is_valid_move = True
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_W and 0 <= new_head[1] < self.GRID_H):
            is_valid_move = False
        # Self/Opponent collision
        if new_head in self.red_snake_body or new_head in self.blue_snake_body:
            is_valid_move = False
            
        # --- State Update ---
        ate_apple = False
        if is_valid_move and moved:
            if self.current_player == 'red':
                # Insert intermediate step for red snake
                intermediate_step = (old_head[0] + new_dir[0], old_head[1] + new_dir[1])
                active_snake_body.insert(0, intermediate_step)
                active_snake_body.insert(0, new_head)
                self.red_snake_dir = new_dir
            else:
                active_snake_body.insert(0, new_head)
                self.blue_snake_dir = new_dir

            # Apple consumption
            if new_head in self.apples:
                ate_apple = True
                if self.current_player == 'red': self.red_score += 1
                else: self.blue_score += 1
                self.apples.remove(new_head)
                self._spawn_apple()
                self._create_particles(new_head)
            else:
                active_snake_body.pop() # Move by removing tail
                if self.current_player == 'red':
                    active_snake_body.pop() # Red snake moves 2 squares, so pop twice

        # --- Reward Calculation ---
        reward = self._calculate_reward(active_snake_body, old_head, new_head, ate_apple)
        
        # --- Termination Check ---
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS or not is_valid_move
        if not is_valid_move:
            reward -= 100 # Penalty for invalid move
        
        if terminated:
            self.game_over = True
            # Add terminal reward
            if self.red_score > self.blue_score: reward += 50
            elif self.red_score < self.blue_score: reward -= 50
        
        # --- Switch Player ---
        self.current_player = 'blue' if self.current_player == 'red' else 'red'
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_reward(self, active_snake_body, old_head, new_head, ate_apple):
        reward = 0
        if ate_apple:
            reward += 10
        
        # Distance-based reward
        if self.apples:
            head = new_head
            
            # Find closest apple
            closest_apple = min(self.apples, key=lambda a: math.hypot(a[0] - head[0], a[1] - head[1]))
            old_dist = math.hypot(closest_apple[0] - old_head[0], closest_apple[1] - old_head[1])
            new_dist = math.hypot(closest_apple[0] - head[0], closest_apple[1] - head[1])

            if new_dist < old_dist:
                reward += 1.0 # Moved closer
            else:
                reward -= 0.1 # Moved away or stayed same distance
                
        return reward

    def _spawn_apple(self):
        while True:
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            if self._is_valid_spawn(pos):
                self.apples.append(pos)
                return

    def _is_valid_spawn(self, pos):
        if pos in self.red_snake_body: return False
        if pos in self.blue_snake_body: return False
        if pos in self.apples: return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "red_score": self.red_score,
            "blue_score": self.blue_score,
            "steps": self.steps,
            "turn": (self.steps // 2) + 1
        }

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_particles()
        self._draw_apples()
        self._draw_snakes()
        self._draw_active_indicator()

    def _render_ui(self):
        # Red Score
        red_text = self.font_large.render(f"{self.red_score}", True, self.COLOR_RED)
        self.screen.blit(red_text, (20, 10))
        
        # Blue Score
        blue_text = self.font_large.render(f"{self.blue_score}", True, self.COLOR_BLUE)
        self.screen.blit(blue_text, (self.WIDTH - blue_text.get_width() - 20, 10))

        # Turn indicator
        turn_num = (self.steps // 2) + 1
        turn_text_str = f"TURN {turn_num}" if not self.game_over else "GAME OVER"
        turn_text = self.font_small.render(turn_text_str, True, self.COLOR_TEXT)
        text_rect = turn_text.get_rect(center=(self.WIDTH / 2, 25))
        self.screen.blit(turn_text, text_rect)

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_snakes(self):
        self._draw_single_snake(self.red_snake_body, self.COLOR_RED)
        self._draw_single_snake(self.blue_snake_body, self.COLOR_BLUE)

    def _draw_single_snake(self, body, color):
        if not body: return
        radius = self.GRID_SIZE // 2
        
        for i, segment in enumerate(body):
            pos = (int(segment[0] * self.GRID_SIZE + radius), int(segment[1] * self.GRID_SIZE + radius))
            
            # Draw smooth connections
            if i > 0:
                prev_pos = (int(body[i-1][0] * self.GRID_SIZE + radius), int(body[i-1][1] * self.GRID_SIZE + radius))
                pygame.draw.line(self.screen, color, prev_pos, pos, self.GRID_SIZE)

        for i, segment in enumerate(body):
            pos = (int(segment[0] * self.GRID_SIZE + radius), int(segment[1] * self.GRID_SIZE + radius))
            current_radius = radius if i > 0 else int(radius * 1.2) # Head is larger
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_radius, color)

    def _draw_apples(self):
        radius = self.GRID_SIZE // 2
        for apple in self.apples:
            pos = (int(apple[0] * self.GRID_SIZE + radius), int(apple[1] * self.GRID_SIZE + radius))
            
            # Glow effect
            glow_size = int(radius * (1.5 + 0.2 * math.sin(pygame.time.get_ticks() / 200)))
            glow_alpha = 60
            s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_APPLE_GLOW, glow_alpha), (glow_size, glow_size), glow_size)
            self.screen.blit(s, (pos[0] - glow_size, pos[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Apple circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_APPLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_APPLE)

    def _draw_active_indicator(self):
        if self.game_over: return
        
        active_snake_body = self.red_snake_body if self.current_player == 'red' else self.blue_snake_body
        color = self.COLOR_RED if self.current_player == 'red' else self.COLOR_BLUE
        
        if not active_snake_body: return
        
        head = active_snake_body[0]
        radius = self.GRID_SIZE // 2
        pos = (int(head[0] * self.GRID_SIZE + radius), int(head[1] * self.GRID_SIZE + radius))
        
        # Pulsating circle around active head
        indicator_radius = int(self.GRID_SIZE * (0.8 + 0.1 * math.sin(pygame.time.get_ticks() / 150)))
        alpha = int(100 + 50 * math.sin(pygame.time.get_ticks() / 150))
        
        s = pygame.Surface((indicator_radius * 2, indicator_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (indicator_radius, indicator_radius), indicator_radius, width=2)
        self.screen.blit(s, (pos[0] - indicator_radius, pos[1] - indicator_radius))

    def _create_particles(self, grid_pos):
        radius = self.GRID_SIZE // 2
        pos_x = grid_pos[0] * self.GRID_SIZE + radius
        pos_y = grid_pos[1] * self.GRID_SIZE + radius
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(15, 30)
            self.particles.append([pos_x, pos_y, vx, vy, lifetime, self.COLOR_APPLE])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                size = int(max(0, (p[4] / 30) * 5))
                alpha = int(max(0, (p[4] / 30) * 255))
                color = (*p[5], alpha)
                
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p[0]-size), int(p[1]-size)))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup for human play
    pygame.display.set_caption("Snake Showdown")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    print("--- Human Controls ---")
    print("Arrows: Move")
    print("R: Reset")
    print("Q: Quit")
    print("--------------------")

    while running:
        action = [0, 0, 0] # Default: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                
                # Note: This is a simplified mapping for human play.
                # An agent would provide an action on every step.
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                # Only step if a move key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Step: {info['steps']}, Turn: {info['turn']}, Player: {'Blue' if env.current_player == 'red' else 'Red'}'s turn")
                    print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
                    print(f"Scores: Red {info['red_score']} - Blue {info['blue_score']}")

                    if terminated:
                        print("--- GAME OVER ---")
                        if info['red_score'] > info['blue_score']: print("Red Wins!")
                        elif info['blue_score'] > info['red_score']: print("Blue Wins!")
                        else: print("It's a Tie!")

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()