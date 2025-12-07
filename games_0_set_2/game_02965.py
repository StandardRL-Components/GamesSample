
# Generated: 2025-08-28T06:34:15.818943
# Source Brief: brief_02965.md
# Brief Index: 2965

        
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
        "Controls: ←→ to aim the launcher. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-arcade brick breaker. Aim your shots to clear the board, "
        "but be warned: overly safe shots are penalized. Clear all 75 bricks to win."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (15, 15, 25)
    COLOR_BALL = (255, 255, 255)
    COLOR_LAUNCHER = (180, 180, 190)
    COLOR_TEXT = (220, 220, 220)
    COLOR_AIM_LINE = (255, 255, 255, 100)

    BRICK_ROWS = 5
    BRICK_COLS = 15
    BRICK_HEIGHT = 12
    BRICK_WIDTH = 38
    BRICK_H_SPACING = 4
    BRICK_V_SPACING = 4
    BRICK_START_Y = 40
    BRICK_PALETTE = [
        (227, 89, 79), (242, 133, 76), (247, 180, 81),
        (145, 201, 90), (76, 188, 137), (82, 137, 199),
        (127, 103, 181)
    ]

    LAUNCHER_WIDTH = 100
    LAUNCHER_HEIGHT = 15
    LAUNCHER_Y = SCREEN_HEIGHT - 30
    
    BALL_RADIUS = 5
    BALL_SPEED = 5.5

    MAX_STEPS = 1000
    INITIAL_BALLS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        self.launcher_angle = 0.0
        self.ball_state = 'ready'
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.bricks = []
        self.particles = []
        self.score = 0
        self.balls_left = 0
        self.steps = 0
        self.game_over = False
        self.total_bricks = self.BRICK_ROWS * self.BRICK_COLS
        self.bricks_destroyed = 0
        
        self.launcher_rect = pygame.Rect(
            self.SCREEN_WIDTH / 2 - self.LAUNCHER_WIDTH / 2,
            self.LAUNCHER_Y,
            self.LAUNCHER_WIDTH,
            self.LAUNCHER_HEIGHT
        )
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.bricks_destroyed = 0
        self.launcher_angle = 0.0
        self.particles = []
        
        self._setup_bricks()
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0
        
        if self.ball_state == 'ready':
            # Handle aiming
            if movement == 3:  # Left
                self.launcher_angle = min(80, self.launcher_angle + 4)
            elif movement == 4: # Right
                self.launcher_angle = max(-80, self.launcher_angle - 4)

            # Handle launch
            if space_held:
                # sfx: launch_ball.wav
                self.ball_state = 'moving'
                angle_rad = math.radians(self.launcher_angle - 90)
                self.ball_vel = pygame.Vector2(
                    math.cos(angle_rad) * self.BALL_SPEED,
                    math.sin(angle_rad) * self.BALL_SPEED
                )
        
        # If ball is moving, run the simulation for this turn
        if self.ball_state == 'moving':
            turn_reward, ball_lost = self._simulate_ball_flight()
            reward += turn_reward
            if ball_lost:
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    reward -= 100 # Terminal penalty for losing

        self.steps += 1
        
        # Check for termination conditions
        win = self.bricks_destroyed == self.total_bricks
        loss = self.balls_left <= 0
        timeout = self.steps >= self.MAX_STEPS
        
        terminated = win or loss or timeout
        if win and not self.game_over:
            reward += 100 # Terminal reward for winning
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _simulate_ball_flight(self):
        reward = 0
        ball_lost = False
        
        # This loop simulates physics until the ball is caught or lost
        # A safety break prevents infinite loops
        for _ in range(500): 
            self.ball_pos += self.ball_vel

            # Wall collisions
            if self.ball_pos.x - self.BALL_RADIUS < 0:
                self.ball_pos.x = self.BALL_RADIUS
                self.ball_vel.x *= -1
                # sfx: bounce_wall.wav
            if self.ball_pos.x + self.BALL_RADIUS > self.SCREEN_WIDTH:
                self.ball_pos.x = self.SCREEN_WIDTH - self.BALL_RADIUS
                self.ball_vel.x *= -1
                # sfx: bounce_wall.wav
            if self.ball_pos.y - self.BALL_RADIUS < 0:
                self.ball_pos.y = self.BALL_RADIUS
                self.ball_vel.y *= -1
                # sfx: bounce_wall.wav

            # Ball lost
            if self.ball_pos.y > self.SCREEN_HEIGHT:
                ball_lost = True
                # sfx: ball_lost.wav
                break

            # Launcher collision
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.ball_vel.y > 0 and ball_rect.colliderect(self.launcher_rect):
                self.ball_state = 'ready'
                self._reset_ball() # Resets position and velocity
                # sfx: bounce_paddle.wav

                # Penalty for safe shots
                hit_pos_x = self.ball_pos.x
                launcher_center_x = self.launcher_rect.centerx
                safe_zone_width = self.LAUNCHER_WIDTH * 0.6
                if abs(hit_pos_x - launcher_center_x) < safe_zone_width / 2:
                    reward -= 0.2
                break

            # Brick collisions
            for brick in self.bricks:
                if brick['alive'] and ball_rect.colliderect(brick['rect']):
                    # sfx: brick_hit.wav
                    brick['alive'] = False
                    self.bricks_destroyed += 1
                    self.score += 1
                    reward += 1.1 # +1 for destroy, +0.1 for hit
                    
                    self._spawn_particles(brick['rect'].center, brick['color'])

                    # Bounce logic
                    prev_ball_rect = pygame.Rect(
                        self.ball_pos.x - self.ball_vel.x - self.BALL_RADIUS,
                        self.ball_pos.y - self.ball_vel.y - self.BALL_RADIUS,
                        self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
                    )

                    if prev_ball_rect.bottom <= brick['rect'].top or prev_ball_rect.top >= brick['rect'].bottom:
                        self.ball_vel.y *= -1
                    else:
                        self.ball_vel.x *= -1
                    
                    # Move ball out of collision to prevent sticking
                    self.ball_pos += self.ball_vel 
                    break # Only handle one brick collision per frame

        return reward, ball_lost

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
            "balls_left": self.balls_left,
            "bricks_destroyed": self.bricks_destroyed,
        }
        
    def _setup_bricks(self):
        self.bricks.clear()
        grid_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_H_SPACING) - self.BRICK_H_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick_x = start_x + c * (self.BRICK_WIDTH + self.BRICK_H_SPACING)
                brick_y = self.BRICK_START_Y + r * (self.BRICK_HEIGHT + self.BRICK_V_SPACING)
                color = self.BRICK_PALETTE[(r + c) % len(self.BRICK_PALETTE)]
                self.bricks.append({
                    'rect': pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                    'color': color,
                    'alive': True
                })

    def _reset_ball(self):
        self.ball_state = 'ready'
        self.ball_pos = pygame.Vector2(self.launcher_rect.centerx, self.launcher_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _render_game(self):
        # Bricks
        for brick in self.bricks:
            if brick['alive']:
                pygame.draw.rect(self.screen, brick['color'], brick['rect'], border_radius=2)

        # Launcher
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, self.launcher_rect, border_radius=3)
        
        # Aim line
        if self.ball_state == 'ready':
            angle_rad = math.radians(self.launcher_angle - 90)
            end_x = self.ball_pos.x + 40 * math.cos(angle_rad)
            end_y = self.ball_pos.y + 40 * math.sin(angle_rad)
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, self.ball_pos, (end_x, end_y))

        # Particles
        self._update_and_render_particles()

        # Ball
        if self.balls_left > 0:
            pos = (int(self.ball_pos.x), int(self.ball_pos.y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            pos = (self.SCREEN_WIDTH - 20 - i * 20, 20)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        if self.game_over:
            msg = "YOU WIN!" if self.bricks_destroyed == self.total_bricks else "GAME OVER"
            color = (100, 255, 100) if self.bricks_destroyed == self.total_bricks else (255, 100, 100)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _spawn_particles(self, pos, color):
        for _ in range(8):
            vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': random.randint(15, 30),
                'max_lifespan': 30,
                'color': color,
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                life_ratio = p['lifespan'] / p['max_lifespan']
                size = max(0, int(3 * life_ratio))
                if size > 0:
                    rect = pygame.Rect(p['pos'].x - size/2, p['pos'].y - size/2, size, size)
                    pygame.draw.rect(self.screen, p['color'], rect)

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
    # Note: The environment is designed for agent interaction, not human play.
    # This loop is a basic demonstration.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Brick Breaker Gym Env")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        space = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
        
        if terminated:
            # On game over, only listen for reset or quit
            continue

        # --- Environment Step ---
        action = [movement, space, 0] # Movement, Space, Shift
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it.
        # Pygame uses (width, height), but numpy array is (height, width, channels).
        # We need to transpose back for display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Control the human-play frame rate

    env.close()