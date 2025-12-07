
# Generated: 2025-08-27T21:58:28.197341
# Source Brief: brief_02968.md
# Brief Index: 2968

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive the onslaught of descending balls by deflecting them with your paddle. The game gets faster and more balls are added over time. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Visuals
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 50, 50)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (200, 200, 220)

        # Paddle properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8

        # Ball properties
        self.BALL_BASE_RADIUS = 8
        self.INITIAL_BALL_SPEED = 2.0
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables are initialized in reset()
        self.paddle = None
        self.balls = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.ball_base_speed = None
        
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.ball_base_speed = self.INITIAL_BALL_SPEED
        
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        if self.game_over:
            # If the game is over, no updates, just return the final state
            reward = 0
            terminated = True
        else:
            # Game is running
            self.steps += 1
            reward = 0.01  # Survival reward per frame

            # 1. Handle player input and calculate movement penalty
            paddle_moved = self._handle_input(movement)
            if paddle_moved:
                is_ball_near = any(
                    abs(ball['pos'][1] - self.paddle.y) < 20 for ball in self.balls
                )
                if not is_ball_near:
                    reward -= 0.02

            # 2. Update game entities and get event rewards
            deflection_count, game_lost = self._update_balls()
            self._update_particles()
            
            if deflection_count > 0:
                reward += deflection_count * 1.0  # Reward for each deflection
                self.score += deflection_count
            
            if game_lost:
                self.game_over = True
                self.win = False
                # SFX: Game Over sound

            # 3. Handle difficulty progression
            self._scale_difficulty()
            
            # 4. Check for termination conditions and apply terminal rewards
            terminated = False
            if self.game_over:
                reward -= 10  # Penalty for losing
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.win = True
                reward += 100  # Large reward for winning
                terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        # Clamp paddle to screen bounds
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))
        return moved

    def _update_balls(self):
        deflection_count = 0
        game_lost = False

        for ball in self.balls:
            ball['pos'] += ball['vel']
            
            current_radius = self._get_current_ball_radius()

            # Wall collisions
            if ball['pos'][0] <= current_radius or ball['pos'][0] >= self.WIDTH - current_radius:
                ball['vel'][0] *= -1
                ball['pos'][0] = np.clip(ball['pos'][0], current_radius, self.WIDTH - current_radius)
                # SFX: Wall bounce
            if ball['pos'][1] <= current_radius:
                ball['vel'][1] *= -1
                ball['pos'][1] = np.clip(ball['pos'][1], current_radius, self.HEIGHT)
                # SFX: Wall bounce

            # Paddle collision
            ball_rect = pygame.Rect(ball['pos'][0] - current_radius, ball['pos'][1] - current_radius, current_radius * 2, current_radius * 2)
            if self.paddle.colliderect(ball_rect) and ball['vel'][1] > 0:
                # SFX: Paddle hit
                
                # Prevent ball from getting stuck in paddle
                ball['pos'][1] = self.paddle.top - current_radius
                
                # Calculate bounce angle based on impact point
                offset = (ball['pos'][0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                
                # Reverse vertical velocity and add horizontal influence
                ball['vel'][1] *= -1
                ball['vel'][0] += offset * 2.5
                
                # Normalize speed to prevent runaway velocity
                speed = np.linalg.norm(ball['vel'])
                if speed > 0:
                    ball['vel'] = (ball['vel'] / speed) * self.ball_base_speed

                deflection_count += 1
                self._create_particles(ball['pos'].copy(), 20)

            # Bottom edge (loss condition)
            if ball['pos'][1] >= self.HEIGHT:
                game_lost = True
        
        return deflection_count, game_lost

    def _scale_difficulty(self):
        # Increase speed every 10 seconds (600 frames)
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.ball_base_speed += 0.5
            # SFX: Speed up chime

        # Add a new ball every 30 seconds (1800 frames), up to a max of 2
        if self.steps > 0 and self.steps % (30 * self.FPS) == 0 and len(self.balls) < 2:
            self._spawn_ball()
            # SFX: New ball spawn

    def _spawn_ball(self):
        spawn_x = self.np_random.uniform(self.BALL_BASE_RADIUS, self.WIDTH - self.BALL_BASE_RADIUS)
        spawn_y = self.BALL_BASE_RADIUS + 5
        
        angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        vel_x = math.cos(angle) * self.ball_base_speed
        vel_y = math.sin(angle) * self.ball_base_speed
        
        new_ball = {
            'pos': np.array([spawn_x, spawn_y], dtype=float),
            'vel': np.array([vel_x, vel_y], dtype=float)
        }
        self.balls.append(new_ball)

    def _create_particles(self, position, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(10, 25)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9  # Friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (0, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 25))
            color = (*self.COLOR_PARTICLE, alpha)
            size = max(1, int(p['life'] / 8))
            pygame.draw.circle(self.screen, color, p['pos'].astype(int), size)

        # Draw balls
        current_radius = self._get_current_ball_radius()
        for ball in self.balls:
            pos_int = ball['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(current_radius), self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(current_radius), self.COLOR_BALL)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
    def _render_ui(self):
        # Render Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Render Timer
        time_remaining = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        timer_text = f"TIME: {time_remaining}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Render Game Over/Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (50, 255, 50) if self.win else (255, 50, 50)
            msg_surf = self.font_game_over.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)
            
    def _get_current_ball_radius(self):
        # Ball size increases with speed
        return self.BALL_BASE_RADIUS + (self.ball_base_speed - self.INITIAL_BALL_SPEED) * 1.5

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "ball_speed": self.ball_base_speed,
            "num_balls": len(self.balls)
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
    env = GameEnv()
    env.reset()
    
    # Use Pygame for human interaction
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ball Deflector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        # Map keyboard inputs to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            env.reset()
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()