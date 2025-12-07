
# Generated: 2025-08-27T21:58:46.588870
# Source Brief: brief_02976.md
# Brief Index: 2976

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A visually stunning, procedurally generated Breakout-style arcade game.
    The player controls a paddle to bounce a ball and break bricks for points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the bricks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball with a paddle to break bricks and achieve a high score in a "
        "visually stunning, procedurally generated arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    WIN_SCORE = 500
    MAX_LIVES = 3

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 200, 0)
    COLOR_TEXT = (255, 255, 255)
    BRICK_COLORS = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), 
        (255, 128, 0), (255, 255, 0)
    ]

    # Game Object Properties
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED = 6
    BRICK_ROWS, BRICK_COLS = 5, 12
    BRICK_WIDTH, BRICK_HEIGHT = 50, 20
    BRICK_GAP = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.win = False

        self.reset()
        
        # This check is for development and ensures API compliance
        # self.validate_implementation() 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False

        # Paddle
        self.paddle = pygame.Rect(
            self.WIDTH / 2 - self.PADDLE_WIDTH / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
        
        # Bricks
        self._create_bricks()

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def _create_bricks(self):
        self.bricks = []
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP) - self.BRICK_GAP
        start_x = (self.WIDTH - total_brick_width) / 2
        start_y = 50
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                x = start_x + j * (self.BRICK_WIDTH + self.BRICK_GAP)
                y = start_y + i * (self.BRICK_HEIGHT + self.BRICK_GAP)
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                brick = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append({"rect": brick, "color": color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement = action[0]

        # 1. Handle player input
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        if moved:
            reward -= 0.02

        # 2. Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # 3. Collision detection
        # Walls
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            # Influence ball direction based on where it hits the paddle
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.BALL_SPEED * 1.2
            # Ensure vertical speed is maintained
            speed_sq = self.ball_vel[0]**2 + self.ball_vel[1]**2
            desired_speed_sq = self.BALL_SPEED**2
            if speed_sq > desired_speed_sq:
                 self.ball_vel[0] /= math.sqrt(speed_sq / desired_speed_sq)
                 self.ball_vel[1] /= math.sqrt(speed_sq / desired_speed_sq)

            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            reward += 0.1
            # sfx: paddle_hit

        # Bricks
        hit_brick = None
        for brick_data in self.bricks:
            if ball_rect.colliderect(brick_data["rect"]):
                hit_brick = brick_data
                break
        
        if hit_brick:
            self.bricks.remove(hit_brick)
            self.ball_vel[1] *= -1 # Simple bounce
            self.score += 1
            reward += 1
            self._create_particles(hit_brick["rect"].center, hit_brick["color"])
            # sfx: brick_break

        # Bottom wall (lose life)
        if self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.lives -= 1
            reward -= 1
            # sfx: lose_life
            if self.lives > 0:
                # Reset ball
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
            else:
                self.game_over = True

        # 4. Update particles
        self._update_particles()
        
        # 5. Check termination conditions
        self.steps += 1
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
            reward += 100
        elif not self.bricks: # All bricks broken
            self.win = True
            self.game_over = True
            reward += 100 # Win even if score is not 500
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _render_game(self):
        # Background gradient
        self._draw_gradient_background()
        
        # Bricks
        for brick_data in self.bricks:
            pygame.draw.rect(self.screen, brick_data["color"], brick_data["rect"])

        # Paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_rect.center = self.paddle.center
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Ball with glow
        self._draw_glowing_circle(
            self.screen, self.COLOR_BALL, self.COLOR_BALL_GLOW,
            (int(self.ball_pos[0]), int(self.ball_pos[1])), self.BALL_RADIUS, 5
        )

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

        # UI
        self._render_ui()

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_gradient_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
    
    def _draw_glowing_circle(self, surface, color, glow_color, center, radius, glow_width):
        # Draw multiple semi-transparent circles to create a glow effect
        for i in range(glow_width, 0, -1):
            alpha = int(150 * (1 - (i / glow_width)))
            glow_surf = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, alpha), (radius + i, radius + i), radius + i)
            surface.blit(glow_surf, (center[0] - radius - i, center[1] - radius - i))
        # Draw the main circle on top
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.font.quit()
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


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # For human playback, we need a visible display
        import os
        os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac' etc.
        
    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()

    if render_mode == "human":
        # Create a display for human viewing
        pygame.display.set_caption("Breakout Arcade")
        human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    total_reward = 0
    
    # --- Human Controls ---
    # We map keyboard keys to the MultiDiscrete action space
    keys_to_action = {
        pygame.K_LEFT: [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- RESET ---")
            
            # Continuous key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = keys_to_action[pygame.K_LEFT]
            elif keys[pygame.K_RIGHT]:
                action = keys_to_action[pygame.K_RIGHT]
        else: # Random agent for rgb_array mode
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # Blit the environment's internal screen to the human-visible screen
            # Need to transpose the observation back to pygame's (width, height, channels) format
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Control FPS for human play
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            if render_mode == "human":
                # Wait for a moment before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
            else:
                running = False # End loop for automated testing
    
    env.close()