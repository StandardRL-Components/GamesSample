
# Generated: 2025-08-28T04:04:14.067301
# Source Brief: brief_02209.md
# Brief Index: 2209

        
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
        "Controls: ←→ to move the paddle. Hold shift for a temporary paddle boost. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, procedurally generated block breaker. Destroy falling blocks with the ball before they reach the bottom."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.np_random = None

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED = 8
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 60, 20
        self.BLOCK_COLS = 10
        self.BLOCK_GAP = 4
        self.INITIAL_LIVES = 3
        self.WIN_CONDITION_LINES = 20
        self.MAX_STEPS = 10000
        self.PADDLE_BOOST_DURATION = 30 # steps
        self.PADDLE_BOOST_COOLDOWN = 90 # steps

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.BLOCK_DEFINITIONS = {
            1: {'color': (0, 150, 80), 'points': 1},
            5: {'color': (0, 100, 200), 'points': 5},
            10: {'color': (200, 50, 50), 'points': 10},
        }

        # Fonts
        self.ui_font = pygame.font.SysFont('consolas', 20, bold=True)
        self.game_over_font = pygame.font.SysFont('consolas', 48, bold=True)

        # Rewards
        self.REWARD_HIT_BLOCK = 0.1
        self.REWARD_PADDLE_MOVE = -0.01
        self.REWARD_CLEAR_LINE = 1.0
        self.REWARD_LOSE_LIFE = -1.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # State variables (initialized in reset)
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.lines_cleared = None
        self.game_over = None
        self.block_fall_speed = None
        self.paddle_boost_timer = None
        self.paddle_boost_cooldown = None
        self.next_line_y_trigger = None
        self.next_line_id = None
        self.active_line_ids = None
        
        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_rect = pygame.Rect(
            (self.screen_width - self.PADDLE_WIDTH) / 2,
            self.screen_height - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()

        self.blocks = []
        self.particles = []
        self.active_line_ids = set()
        
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.lines_cleared = 0
        self.game_over = False
        self.block_fall_speed = 0.25
        self.paddle_boost_timer = 0
        self.paddle_boost_cooldown = 0

        self.next_line_id = 0
        self.next_line_y_trigger = 20
        self._generate_initial_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input & State Updates
        reward += self._handle_input(action)
        self._update_paddle_boost()
        
        # 2. Update Game Logic
        block_hit_reward, destroyed_blocks_info = self._update_ball()
        reward += block_hit_reward
        
        life_lost, blocks_at_bottom = self._update_blocks()
        if life_lost:
            reward += self.REWARD_LOSE_LIFE
            self.lives -= 1
            if self.lives > 0:
                self._reset_level(blocks_at_bottom)
            else:
                self.game_over = True

        # 3. Check for cleared lines and update difficulty
        cleared_line_reward = self._check_cleared_lines(destroyed_blocks_info)
        reward += cleared_line_reward
        
        self._generate_new_lines()
        self._update_particles()
        
        # 4. Check Termination
        terminated = self.lives <= 0 or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                reward += self.REWARD_WIN
            elif self.lives <= 0:
                reward += self.REWARD_LOSE
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Paddle Movement
        if movement in [3, 4]:
            reward += self.REWARD_PADDLE_MOVE
            if movement == 3:  # Left
                self.paddle_rect.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle_rect.x += self.PADDLE_SPEED
            self.paddle_rect.left = max(0, self.paddle_rect.left)
            self.paddle_rect.right = min(self.screen_width, self.paddle_rect.right)

        # Ball Launch
        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.math.Vector2(
                self.BALL_MAX_SPEED * math.sin(angle),
                -self.BALL_MAX_SPEED * math.cos(angle) * 0.8
            )

        # Paddle Boost
        if shift_held and self.paddle_boost_cooldown == 0:
            self.paddle_boost_timer = self.PADDLE_BOOST_DURATION
            self.paddle_boost_cooldown = self.PADDLE_BOOST_COOLDOWN
        
        return reward

    def _update_paddle_boost(self):
        if self.paddle_boost_cooldown > 0:
            self.paddle_boost_cooldown -= 1
        
        if self.paddle_boost_timer > 0:
            self.paddle_boost_timer -= 1
            self.paddle_rect.width = self.PADDLE_WIDTH * 1.5
        else:
            self.paddle_rect.width = self.PADDLE_WIDTH

    def _update_ball(self):
        if not self.ball_launched:
            self.ball_pos.x = self.paddle_rect.centerx
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
            return 0, []

        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.screen_width - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.screen_width - self.BALL_RADIUS, self.ball_pos.x))

        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
        
        # Bounce off bottom wall (as per brief, only blocks reaching bottom cause life loss)
        if self.ball_pos.y >= self.screen_height - self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.screen_height - self.BALL_RADIUS

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            # Sound: paddle_hit.wav
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
            self.ball_vel.y *= -1
            
            offset = (self.ball_pos.x - self.paddle_rect.centerx) / (self.paddle_rect.width / 2)
            self.ball_vel.x = self.BALL_MAX_SPEED * offset
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_MAX_SPEED

        # Anti-stuck mechanism
        if abs(self.ball_vel.y) < 2:
            self.ball_vel.y = math.copysign(2, self.ball_vel.y)

        # Block collisions
        reward = 0
        destroyed_blocks = []
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if ball_rect.colliderect(block['rect']):
                # Sound: block_break.wav
                reward += self.REWARD_HIT_BLOCK
                self.score += block['points']
                
                # Determine collision side for realistic bounce
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1

                self._create_particles(block['rect'].center, block['color'])
                destroyed_blocks.append(block)
                self.blocks.pop(i)
                break # Only handle one collision per frame
        
        return reward, destroyed_blocks

    def _update_blocks(self):
        life_lost = False
        blocks_at_bottom = []
        for block in self.blocks:
            block['rect'].y += self.block_fall_speed
            if block['rect'].bottom > self.screen_height:
                life_lost = True
                blocks_at_bottom.append(block)
        
        if life_lost:
            # Sound: life_lost.wav
            for block in blocks_at_bottom:
                if block in self.blocks:
                    self.blocks.remove(block)
                    if block['line_id'] in self.active_line_ids:
                        self.active_line_ids.remove(block['line_id'])
        
        return life_lost, blocks_at_bottom

    def _check_cleared_lines(self, destroyed_blocks_info):
        if not destroyed_blocks_info:
            return 0
        
        reward = 0
        line_ids_to_check = {b['line_id'] for b in destroyed_blocks_info}
        
        for line_id in line_ids_to_check:
            if line_id in self.active_line_ids:
                is_line_present = any(b['line_id'] == line_id for b in self.blocks)
                if not is_line_present:
                    # Sound: line_clear.wav
                    self.lines_cleared += 1
                    reward += self.REWARD_CLEAR_LINE
                    self.active_line_ids.remove(line_id)
                    
                    # Difficulty scaling
                    if self.lines_cleared > 0 and self.lines_cleared % 2 == 0:
                        self.block_fall_speed += 0.05
        return reward

    def _generate_new_lines(self):
        highest_y = 0
        if self.blocks:
            highest_y = min(b['rect'].top for b in self.blocks)

        if not self.blocks or highest_y > self.next_line_y_trigger:
            self._generate_line(y_pos=-self.BLOCK_HEIGHT)
            self.next_line_y_trigger = 0

    def _generate_line(self, y_pos):
        line_id = self.next_line_id
        self.next_line_id += 1
        self.active_line_ids.add(line_id)
        
        total_block_width = self.BLOCK_COLS * self.BLOCK_WIDTH + (self.BLOCK_COLS - 1) * self.BLOCK_GAP
        start_x = (self.screen_width - total_block_width) / 2
        
        for i in range(self.BLOCK_COLS):
            if self.np_random.random() < 0.8:  # 80% chance for a block to appear
                x = start_x + i * (self.BLOCK_WIDTH + self.BLOCK_GAP)
                
                # Choose block type based on probability
                rand_val = self.np_random.random()
                if rand_val < 0.1:
                    block_type = 10
                elif rand_val < 0.3:
                    block_type = 5
                else:
                    block_type = 1
                
                block_def = self.BLOCK_DEFINITIONS[block_type]
                self.blocks.append({
                    'rect': pygame.Rect(x, y_pos, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    'color': block_def['color'],
                    'points': block_def['points'],
                    'line_id': line_id
                })

    def _generate_initial_blocks(self):
        for i in range(4):
            self._generate_line(y_pos=20 + i * (self.BLOCK_HEIGHT + 20))

    def _reset_level(self, blocks_at_bottom):
        self._reset_ball()
        self.paddle_rect.centerx = self.screen_width / 2
        for block in blocks_at_bottom:
             self._create_particles(block['rect'].center, (100,0,0), 20)
        self.blocks.clear()
        self.active_line_ids.clear()
        self.next_line_y_trigger = 20
        self._generate_initial_blocks()

    def _reset_ball(self):
        self.ball_pos = pygame.math.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.ball_launched = False

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.math.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_blocks()
        self._render_paddle()
        self._render_particles()
        self._render_ball()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "lines_cleared": self.lines_cleared,
            "fall_speed": self.block_fall_speed
        }

    def _render_background(self):
        for x in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            # Add a slight 3D effect
            darker_color = tuple(max(0, c - 30) for c in block['color'])
            pygame.draw.rect(self.screen, darker_color, block['rect'].inflate(-6,-6), border_radius=2)


    def _render_paddle(self):
        color = (255, 255, 100) if self.paddle_boost_timer > 0 else self.COLOR_PADDLE
        pygame.draw.rect(self.screen, color, self.paddle_rect, border_radius=5)
        # Glow effect for boost
        if self.paddle_boost_timer > 0:
            glow_rect = self.paddle_rect.inflate(8, 8)
            alpha = int(100 * (self.paddle_boost_timer / self.PADDLE_BOOST_DURATION))
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

    def _render_ball(self):
        pos = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, *pos, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *pos, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Glow effect
        for i in range(3):
            alpha = 60 - i * 20
            pygame.gfxdraw.aacircle(self.screen, *pos, self.BALL_RADIUS + i + 1, (*self.COLOR_BALL, alpha))

    def _render_particles(self):
        for p in self.particles:
            size = int(max(1, (p['lifespan'] / 30) * 4))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), size, size))

    def _render_ui(self):
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        lines_text = self.ui_font.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (self.screen_width // 2 - lines_text.get_width() // 2, 5))

        lives_text = self.ui_font.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.screen_width - lives_text.get_width() - 10, 5))

        if self.game_over:
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            over_text = self.game_over_font.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-closing or allow reset
            pygame.time.wait(3000)
            running = False # or reset the game: obs, info = env.reset(); total_reward = 0
            
        clock.tick(30) # Match the auto_advance rate
        
    env.close()