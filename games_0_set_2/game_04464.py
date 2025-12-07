import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set SDL_VIDEODRIVER to dummy for headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    """
    A minimalist Breakout-style game implemented as a Gymnasium environment.

    The agent controls a paddle at the bottom of the screen to bounce a ball
    and destroy a grid of blocks at the top. The goal is to destroy all
    blocks without losing all three lives.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle."
    )

    # Short, user-facing description of the game
    game_description = (
        "Minimalist Breakout. Control the paddle to bounce the ball and destroy all the blocks."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed frame rate for smooth visuals
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_WALL = (100, 100, 100)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_LIVES = (220, 50, 50)
        self.COLOR_TEXT = (200, 200, 200)
        self.BLOCK_COLORS = {
            1: (60, 120, 220),  # Blue
            2: (60, 220, 120),  # Green
            3: (220, 220, 60),  # Yellow
        }

        # Game element properties
        self.WALL_THICKNESS = 10
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED = 10
        self.BALL_MIN_VY = 4

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
        self.font_score = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle_x = 0
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.blocks = []
        self.particles = []
        self.ball_trail = []
        self.terminated = False
        self.game_over_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.terminated = False

        # Paddle
        self.paddle_x = self.WIDTH / 2

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        num_cols, num_rows = 10, 5
        block_w = (self.WIDTH - 2 * self.WALL_THICKNESS) / num_cols
        block_h = 20
        y_offset = 50
        
        for r in range(num_rows):
            for c in range(num_cols):
                points = (r // 2) + 1  # Rows 0-1: 1pt, 2-3: 2pt, 4: 3pt
                color = self.BLOCK_COLORS[points]
                block_rect = pygame.Rect(
                    self.WALL_THICKNESS + c * block_w,
                    y_offset + r * block_h,
                    block_w,
                    block_h
                )
                self.blocks.append({
                    "rect": block_rect,
                    "color": color,
                    "points": points,
                    "alive": True
                })

        # Effects
        self.particles = []
        self.ball_trail = []

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 20], dtype=float)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards angle
        speed = 7
        self.ball_vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
        self.ball_trail = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        reward = 0
        
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement
        elif movement == 4:  # Right
            self.paddle_x += self.PADDLE_SPEED
            reward -= 0.01

        # Clamp paddle position
        paddle_min_x = self.WALL_THICKNESS + self.PADDLE_WIDTH / 2
        paddle_max_x = self.WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH / 2
        self.paddle_x = np.clip(self.paddle_x, paddle_min_x, paddle_max_x)
        
        # --- Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # --- Collision Detection ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- State Updates ---
        self.steps += 1
        
        # --- Termination Conditions ---
        all_blocks_destroyed = not any(b['alive'] for b in self.blocks)
        max_steps_reached = self.steps >= self.MAX_STEPS
        
        if self.lives <= 0:
            self.terminated = True
            reward -= 50  # Terminal penalty for losing
            self.game_over_message = "GAME OVER"
        elif all_blocks_destroyed:
            self.terminated = True
            reward += 100 # Terminal reward for winning
            self.game_over_message = "YOU WIN!"
        elif max_steps_reached:
            # Using terminated instead of truncated for consistency with original logic
            self.terminated = True 
        
        if self.terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False, # truncated is always False
            self._get_info()
        )

    def _update_ball(self):
        # Update trail
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 5:
            self.ball_trail.pop(0)

        self.ball_pos += self.ball_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WALL_THICKNESS + self.BALL_RADIUS
        if ball_rect.right >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.WALL_THICKNESS + self.BALL_RADIUS

        # Bottom (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball()
            return reward

        # Paddle
        paddle_rect = pygame.Rect(self.paddle_x - self.PADDLE_WIDTH / 2, self.HEIGHT - self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
            self.ball_pos[1] = self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1
            self.ball_vel[1] *= -1
            
            offset = (self.ball_pos[0] - self.paddle_x) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = np.clip(offset * self.BALL_MAX_SPEED, -self.BALL_MAX_SPEED, self.BALL_MAX_SPEED)
            
            self.ball_vel[1] = -max(abs(self.ball_vel[1]), self.BALL_MIN_VY)

            reward += 0.1
            self._create_particles(self.ball_pos, self.COLOR_PADDLE, 20)
            return reward

        # Blocks
        for block in self.blocks:
            if block['alive'] and ball_rect.colliderect(block['rect']):
                block['alive'] = False
                reward += block['points']
                self.score += block['points']

                self.ball_vel[1] *= -1
                
                self._create_particles(block['rect'].center, block['color'], 30)
                return reward
        
        return reward

    def _create_particles(self, pos, color, count):
        # Ensure pos is a numpy array for vector operations, handling tuples from pygame.Rect.center
        start_pos = np.array(pos, dtype=float)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': start_pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Blocks
        for block in self.blocks:
            if block['alive']:
                pygame.draw.rect(self.screen, block['color'], block['rect'])
                pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            except TypeError: # Handle potential color with alpha issues in some pygame versions
                pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))


        # Ball Trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(100 * (i / len(self.ball_trail)))
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, (*self.COLOR_BALL, alpha))
            except TypeError:
                pass # Ignore if alpha drawing fails

        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Paddle
        paddle_rect = pygame.Rect(0, 0, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        paddle_rect.center = (self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT / 2)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_surf = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))

        # Lives
        for i in range(self.lives):
            pos = (self.WIDTH - self.WALL_THICKNESS - 20 - (i * 30), self.WALL_THICKNESS + 18)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_LIVES)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_LIVES)

        # Game Over Message
        if self.game_over:
            msg_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    
    env = GameEnv(render_mode="rgb_array")
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode terminated.")
            break
            
    # To visualize the game, you would need a display
    # Example for local rendering (requires a display):
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        # Unset the dummy driver to allow display
        del os.environ["SDL_VIDEODRIVER"]
        pygame.quit() # Quit the headless instance
        pygame.init() # Re-init for display

        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Breakout")
        
        running = True
        total_reward = 0
        
        while running:
            action = [0, 0, 0] # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Draw the observation to the screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()
                total_reward = 0

            env.clock.tick(env.FPS)
            
        env.close()