
# Generated: 2025-08-28T04:51:39.654005
# Source Brief: brief_02449.md
# Brief Index: 2449

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, top-down block-breaking game where strategic paddle positioning and risk-taking are rewarded."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (180, 180, 180)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = {
        1: (76, 175, 80),   # Green
        2: (33, 150, 243),  # Blue
        3: (244, 67, 54),   # Red
    }
    BLOCK_SCORES = {1: 1, 2: 2, 3: 3}

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 12
    BALL_RADIUS = 6
    BALL_BASE_SPEED = 6.0
    MAX_BALLS = 3
    TIME_PER_STAGE = 60  # seconds
    MAX_STEPS = TIME_PER_STAGE * FPS * 3 + 100 # Max steps for 3 stages + buffer

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_remaining = None
        self.current_stage = None
        self.time_remaining = None
        self.ball_speed_multiplier = None
        self.prev_space_held = None

        self.reset()

        # self.validate_implementation() # Optional: call to check implementation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = self.MAX_BALLS
        self.current_stage = 1
        self.ball_speed_multiplier = 1.0
        
        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._setup_stage()

        self.particles = []
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        # Action feedback reward
        if movement in [3, 4]: # left/right
            reward -= 0.01 # Small penalty for movement to encourage efficiency

        # Paddle movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.left = max(0, self.paddle_rect.left)
        self.paddle_rect.right = min(self.SCREEN_WIDTH, self.paddle_rect.right)

        # Launch ball
        if space_held and not self.prev_space_held and self.ball_attached:
            self.ball_attached = False
            initial_angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            speed = self.BALL_BASE_SPEED * self.ball_speed_multiplier
            self.ball_vel = pygame.Vector2(speed * math.sin(initial_angle), -speed * math.cos(initial_angle))
            # Sound: Ball launch
        self.prev_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        self.time_remaining -= 1

        if self.ball_attached:
            self.ball_pos.x = self.paddle_rect.centerx
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel

        # --- Handle Collisions ---
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            self.ball_pos.x = ball_rect.centerx
            # Sound: Wall bounce
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos.y = ball_rect.centery
            # Sound: Wall bounce

        # Paddle collision
        if not self.ball_attached and ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            offset = ball_rect.centerx - self.paddle_rect.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            
            max_bounce_angle = math.pi / 2.5  # 72 degrees
            angle = normalized_offset * max_bounce_angle
            
            speed = self.ball_vel.length()
            self.ball_vel.x = speed * math.sin(angle)
            self.ball_vel.y = -speed * math.cos(angle)

            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS # Prevent sticking
            reward += 0.1 # Reward for hitting the ball
            # Sound: Paddle hit

        # Block collisions
        collided_block_index = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if collided_block_index != -1:
            block_data = self.blocks.pop(collided_block_index)
            block_rect = block_data['rect']
            block_type = block_data['type']
            
            # Add reward and score
            block_reward = self.BLOCK_SCORES[block_type]
            reward += block_reward
            self.score += block_reward * 10

            # Create particles
            self._create_particles(block_rect.center, self.BLOCK_COLORS[block_type])
            
            # Collision response
            prev_ball_rect = pygame.Rect(ball_rect)
            prev_ball_rect.center -= self.ball_vel
            
            if prev_ball_rect.bottom <= block_rect.top or prev_ball_rect.top >= block_rect.bottom:
                 self.ball_vel.y *= -1
            if prev_ball_rect.right <= block_rect.left or prev_ball_rect.left >= block_rect.right:
                 self.ball_vel.x *= -1

            # Sound: Block break
            
        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Check Game Conditions ---
        # Ball lost
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.balls_remaining -= 1
            reward -= 50 # Penalty for losing a ball
            if self.balls_remaining > 0:
                self._reset_ball()
                # Sound: Ball lost
            else:
                self.game_over = True
                terminated = True
                reward -= 100 # Game over penalty
                # Sound: Game over

        # Time out
        if self.time_remaining <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Game over penalty
            # Sound: Game over

        # Stage clear
        if not self.blocks:
            self.current_stage += 1
            reward += 50 # Stage clear reward
            if self.current_stage > 3:
                self.game_over = True
                terminated = True
                reward += 100 # Win game reward
                # Sound: Game win
            else:
                self.ball_speed_multiplier += 0.1 # Increase speed
                self._setup_stage()
                self._reset_ball()
                # Sound: Stage clear

        # Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = pygame.Vector2(
            self.paddle_rect.centerx,
            self.paddle_rect.top - self.BALL_RADIUS
        )
        self.ball_vel = pygame.Vector2(0, 0)

    def _setup_stage(self):
        self.blocks = []
        self.time_remaining = self.TIME_PER_STAGE * self.FPS
        
        block_width = 58
        block_height = 20
        gap = 4
        rows, cols = 5, 10
        start_x = (self.SCREEN_WIDTH - (cols * (block_width + gap) - gap)) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                block_type = 1
                if r < 1: block_type = 3 # Top row is red
                elif r < 3: block_type = 2 # Next two are blue
                
                # Vary layouts per stage
                if self.current_stage == 2 and (c < 2 or c > 7): continue
                if self.current_stage == 3 and (c+r) % 2 != 0: continue

                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                rect = pygame.Rect(int(x), int(y), block_width, block_height)
                self.blocks.append({'rect': rect, 'type': block_type})

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[block['type']]
            pygame.draw.rect(self.screen, color, block['rect'], border_radius=3)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=4)
        
        # Render ball
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            life_ratio = p['life'] / 20.0
            current_radius = int(p['radius'] * life_ratio)
            if current_radius > 0:
                alpha = int(255 * life_ratio)
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (current_radius, current_radius), current_radius)
                self.screen.blit(temp_surf, (pos_int[0] - current_radius, pos_int[1] - current_radius))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_main.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_TEXT)
        balls_rect = balls_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(balls_text, balls_rect)
        
        # Time
        time_seconds = math.ceil(self.time_remaining / self.FPS)
        time_color = self.COLOR_TEXT if time_seconds > 10 else (255, 100, 100)
        time_text = self.font_main.render(f"{time_seconds}", True, time_color)
        time_rect = time_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over / Win Message
        if self.game_over:
            msg = "GAME OVER" if self.balls_remaining == 0 or self.time_remaining <= 0 else "YOU WIN!"
            msg_text = self.font_main.render(msg, True, (255, 255, 255))
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), msg_rect.inflate(20, 20))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "current_stage": self.current_stage,
            "time_remaining_seconds": math.ceil(self.time_remaining / self.FPS),
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    movement = 0 # 0=none, 3=left, 4=right
    space_held = 0 # 0=released, 1=held
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        else:
            movement = 0

        space_held = 1 if keys[pygame.K_SPACE] else 0
        
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()
    pygame.quit()