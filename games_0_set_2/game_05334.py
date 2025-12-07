
# Generated: 2025-08-28T04:43:47.931810
# Source Brief: brief_05334.md
# Brief Index: 5334

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker. Break all the blocks to win, but don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Constants ---
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_UI = (200, 200, 255)
        
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7.5
        self.MAX_STEPS = 2000
        
        # --- Fonts ---
        self.ui_font = pygame.font.Font(None, 28)
        self.end_font = pygame.font.Font(None, 64)

        # --- State (initialized in reset) ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.lives = None
        self.steps = None
        self.game_over = None

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.particles = []

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self._reset_ball()
        self._create_blocks()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.02  # Time penalty to encourage efficiency
        
        self._handle_input(movement, space_held)
        step_reward, life_lost = self._update_game_state()
        reward += step_reward
        if life_lost:
            reward -= 5

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if len(self.blocks) == 0 and self.lives > 0:
                reward += 50  # Win bonus

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        if space_held and not self.ball_launched:
            self.ball_launched = True
            initial_vx = self.np_random.uniform(-1, 1) * self.BALL_SPEED * 0.5
            self.ball_vel = pygame.Vector2(initial_vx, -self.BALL_SPEED).normalize() * self.BALL_SPEED
            # sfx: launch_sound

    def _update_game_state(self):
        reward = 0
        life_lost = False

        if self.ball_launched:
            self.ball_pos += self.ball_vel

            # Anti-softlock: ensure vertical movement
            if abs(self.ball_vel.y) < 1.0:
                self.ball_vel.y = math.copysign(1.5, self.ball_vel.y)

            # Wall collisions
            if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel.x *= -1
                self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
                # sfx: wall_bounce
            if self.ball_pos.y <= self.BALL_RADIUS:
                self.ball_vel.y *= -1
                self.ball_pos.y = self.BALL_RADIUS
                # sfx: wall_bounce

            # Lose life
            if self.ball_pos.y >= self.HEIGHT - self.BALL_RADIUS:
                self.lives -= 1
                life_lost = True
                # sfx: lose_life
                if self.lives > 0:
                    self._reset_ball()

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel.x = np.clip(offset, -1, 1) * self.BALL_SPEED * 0.8
                self.ball_vel.y *= -1
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.BALL_SPEED
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                # sfx: paddle_hit

            # Block collisions
            removed_blocks = []
            for block in self.blocks:
                if block["rect"].colliderect(ball_rect):
                    reward += 1
                    self.score += 1
                    self._create_particles(block["rect"].center, block["color"])
                    
                    # Determine collision side for accurate bounce
                    dx = self.ball_pos.x - block["rect"].centerx
                    dy = self.ball_pos.y - block["rect"].centery
                    w = (self.BALL_RADIUS + block["rect"].width) / 2
                    h = (self.BALL_RADIUS + block["rect"].height) / 2
                    
                    if abs(dx) / w > abs(dy) / h:
                        self.ball_vel.x *= -1
                    else:
                        self.ball_vel.y *= -1
                    
                    removed_blocks.append(block)
                    # sfx: block_break
                    break  # Break one block per frame
            
            if removed_blocks:
                self.blocks = [b for b in self.blocks if b not in removed_blocks]
        else:
            self._reset_ball()

        self._update_particles()
        return reward, life_lost
    
    def _check_termination(self):
        return self.lives <= 0 or len(self.blocks) == 0 or self.steps >= self.MAX_STEPS

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        self.ball_vel = pygame.Vector2(0, 0)
        
    def _create_blocks(self):
        self.blocks = []
        block_colors = [(255, 0, 128), (0, 255, 255), (0, 255, 0), (255, 128, 0), (128, 0, 255)]
        block_w, block_h = 58, 20
        gap = 6
        start_x = (self.WIDTH - (10 * (block_w + gap) - gap)) / 2
        start_y = 50
        for i in range(5):
            for j in range(10):
                x = start_x + j * (block_w + gap)
                y = start_y + i * (block_h + gap)
                rect = pygame.Rect(x, y, block_w, block_h)
                color = block_colors[i % len(block_colors)]
                self.blocks.append({"rect": rect, "color": color})

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Damping
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks with a "glow" effect
        for block in self.blocks:
            glow_rect = block["rect"].inflate(6, 6)
            glow_color = tuple(min(255, c + 50) for c in block["color"])
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*glow_color, 50), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=4)
            pygame.draw.rect(self.screen, (255, 255, 255), block["rect"], 1, border_radius=4)

        # Draw paddle with a "glow" effect
        paddle_glow_rect = self.paddle.inflate(8, 8)
        s = pygame.Surface(paddle_glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_PADDLE, 40), s.get_rect(), border_radius=8)
        self.screen.blit(s, paddle_glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = p["life"] / 30.0
            size = int(alpha * 5)
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], p["pos"], size)
        
        # Draw ball with a "glow" using gfxdraw for antialiasing
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_color = (*self.COLOR_BALL, 100)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.lives):
            pos = (self.WIDTH - 30 - i * 25, 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)
        
        if self.game_over:
            msg = "YOU WIN!" if len(self.blocks) == 0 and self.lives > 0 else "GAME OVER"
            color = (0, 255, 128) if msg == "YOU WIN!" else (255, 0, 100)
            
            end_text = self.end_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # To run with display, you might need to unset the dummy video driver
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control ---
    # To play manually, uncomment this block and comment out the random agent block
    # running = True
    # while running:
    #     movement = 0 # no-op
    #     space = 0
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         movement = 3
    #     if keys[pygame.K_RIGHT]:
    #         movement = 4
    #     if keys[pygame.K_SPACE]:
    #         space = 1
        
    #     action = [movement, space, 0]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
        
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
    #             obs, info = env.reset()
    #             total_reward = 0

    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    #         time.sleep(2) # Pause on game over
    #         obs, info = env.reset()
    #         total_reward = 0
        
    #     env.clock.tick(30) # Run at 30 FPS
    
    # --- Random Agent ---
    for _ in range(3000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            time.sleep(1)

        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        else:
            env.clock.tick(30) # Run at 30 FPS
            continue
        break
        
    env.close()
    pygame.quit()