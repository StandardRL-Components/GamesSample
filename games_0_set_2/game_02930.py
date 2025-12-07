# Generated: 2025-08-27T21:52:01.063351
# Source Brief: brief_02930.md
# Brief Index: 2930

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric-2D block breaker game where strategic paddle positioning and
    risky plays are rewarded. The player must clear three stages of blocks
    without losing all their balls or running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced isometric block breaker. Clear all blocks to advance through "
        "three challenging stages. Don't run out of time or balls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (13, 17, 23)
    COLOR_GRID = (25, 33, 44)
    COLOR_PADDLE = (58, 142, 242)
    COLOR_PADDLE_GLOW = (58, 142, 242, 50)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (255, 255, 255, 100)
    COLOR_BLOCK_STATIC = (225, 83, 99)
    COLOR_BLOCK_MOVING = (210, 153, 34)
    COLOR_BLOCK_INDEST = (100, 108, 122)
    COLOR_TEXT = (230, 237, 243)
    COLOR_PARTICLE = [(255, 83, 99), (255, 120, 130), (255, 180, 180)]

    # Isometric Grid
    ISO_TILE_W = 32
    ISO_TILE_H = 16
    GRID_COLS = 16
    GRID_ROWS = 20
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = UI_HEIGHT + 30

    # Game Mechanics
    PADDLE_GRID_WIDTH = 4
    PADDLE_SPEED = 0.25 # grid units per frame
    BALL_START_SPEED = 2.0 / 60.0 # grid units per frame (scaled for 60fps logic)
    BALL_SPEED_INCREMENT = 0.05 / 60.0
    BALL_RADIUS = 5 # pixels
    MAX_BALLS = 3
    TIME_PER_STAGE = 60 * 30 # 60 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Game state variables
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.balls_left = 0
        self.stage_timer = 0
        
        self.paddle_pos = 0.0
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_in_play = False
        self.blocks_broken_total = 0
        
        self.blocks = []
        self.particles = []
        self.ball_trail = deque(maxlen=8)
        self.ball_y_history = deque(maxlen=150) # For anti-softlock

        # self.reset() is called by the wrapper, but we can call it for standalone use
        # self.reset()

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.ISO_ORIGIN_X + (x - y) * (self.ISO_TILE_W / 2)
        screen_y = self.ISO_ORIGIN_Y + (x + y) * (self.ISO_TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _generate_blocks(self):
        """Generates block layout based on the current stage."""
        self.blocks = []
        rows_to_generate = 8
        
        for r in range(rows_to_generate):
            for c in range(self.GRID_COLS):
                # Create a checkerboard-like pattern that gets denser
                if (r + c) % 2 == 0 and self.np_random.random() < 0.7 + self.stage * 0.1:
                    block_type = 'static'
                    color = self.COLOR_BLOCK_STATIC
                    
                    if self.stage == 2 and self.np_random.random() < 0.25:
                        block_type = 'moving'
                        color = self.COLOR_BLOCK_MOVING
                    elif self.stage == 3 and self.np_random.random() < 0.2:
                        block_type = 'indestructible'
                        color = self.COLOR_BLOCK_INDEST
                    
                    if block_type != 'indestructible' or self.np_random.random() < 0.5:
                         self.blocks.append({
                            'pos': np.array([float(c), float(r)]),
                            'type': block_type,
                            'color': color,
                            'move_dir': 1 if self.np_random.random() > 0.5 else -1,
                            'move_speed': 1.0 / 30.0 # grid units per frame
                        })

    def _setup_stage(self):
        """Initializes state for the current or next stage."""
        self.stage_timer = self.TIME_PER_STAGE
        self.ball_in_play = False
        self.paddle_pos = self.GRID_COLS / 2 - self.PADDLE_GRID_WIDTH / 2
        self.ball_pos = np.array([self.paddle_pos + self.PADDLE_GRID_WIDTH / 2, float(self.GRID_ROWS - 1)])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_trail.clear()
        self.ball_y_history.clear()
        self._generate_blocks()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.balls_left = self.MAX_BALLS
        self.blocks_broken_total = 0
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = -0.01 # Time penalty
        self.steps += 1
        self.stage_timer -= 1
        
        # --- Handle Input ---
        if movement == 3: # Left
            self.paddle_pos -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_pos += self.PADDLE_SPEED
        
        self.paddle_pos = max(0, min(self.GRID_COLS - self.PADDLE_GRID_WIDTH, self.paddle_pos))
        
        if space_pressed and not self.ball_in_play:
            # Launch ball
            self.ball_in_play = True
            # FIX: Swapped low and high arguments. np.random.uniform requires low <= high.
            initial_angle = self.np_random.uniform(-2.2, -0.8) # Upwards angles
            speed = self.BALL_START_SPEED
            self.ball_vel = np.array([math.cos(initial_angle) * speed, math.sin(initial_angle) * speed])
            # sfx: launch_ball.wav

        # --- Update Game State ---
        self._update_particles()

        if self.ball_in_play:
            self._update_moving_blocks()
            collision_reward = self._update_ball_and_collisions()
            reward += collision_reward
        else:
            # Ball follows paddle
            self.ball_pos[0] = self.paddle_pos + self.PADDLE_GRID_WIDTH / 2
        
        # Proximity reward
        if self.ball_in_play and self.ball_pos[1] > self.GRID_ROWS / 2:
            dist_x = abs(self.ball_pos[0] - (self.paddle_pos + self.PADDLE_GRID_WIDTH / 2))
            if dist_x < self.PADDLE_GRID_WIDTH:
                reward += 0.1

        # Check for stage clear
        if self.ball_in_play and not any(b['type'] != 'indestructible' for b in self.blocks):
            self.score += 100
            reward += 10
            self.stage += 1
            if self.stage > 3:
                self.win = True
                self.score += 1000
                reward += 100
            else:
                self._setup_stage()
                # sfx: stage_clear.wav

        # Check termination conditions
        terminated = self.game_over or self.win
        truncated = self.stage_timer <= 0
        if truncated and not self.win:
            self.game_over = True
            terminated = True # Game over also means terminated
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ball_and_collisions(self):
        reward = 0
        
        # Update ball position
        self.ball_pos += self.ball_vel * 30 # Scale to 30fps logic
        self.ball_trail.append(self._iso_to_screen(self.ball_pos[0], self.ball_pos[1]))

        # Anti-softlock
        self.ball_y_history.append(self.ball_pos[1])
        if len(self.ball_y_history) == self.ball_y_history.maxlen:
            if np.std(self.ball_y_history) < 0.01:
                self.ball_vel[1] += (self.np_random.random() - 0.5) * 0.01
                self.ball_vel[0] += (self.np_random.random() - 0.5) * 0.01

        # Wall collisions
        if self.ball_pos[0] < 0:
            self.ball_pos[0] = 0
            self.ball_vel[0] *= -1
        elif self.ball_pos[0] > self.GRID_COLS:
            self.ball_pos[0] = self.GRID_COLS
            self.ball_vel[0] *= -1
        if self.ball_pos[1] < 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] *= -1
        
        # Lose ball
        if self.ball_pos[1] > self.GRID_ROWS:
            self.balls_left -= 1
            reward -= 5
            self.ball_in_play = False
            self.ball_trail.clear()
            self.ball_y_history.clear()
            # Reset ball position to paddle, but don't reset whole stage
            self.paddle_pos = self.GRID_COLS / 2 - self.PADDLE_GRID_WIDTH / 2
            self.ball_pos = np.array([self.paddle_pos + self.PADDLE_GRID_WIDTH / 2, float(self.GRID_ROWS - 1)])
            self.ball_vel = np.array([0.0, 0.0])
            if self.balls_left <= 0:
                self.game_over = True
            # sfx: lose_ball.wav
            return reward

        # Paddle collision
        paddle_y = self.GRID_ROWS - 1
        if self.ball_vel[1] > 0 and paddle_y - 1 < self.ball_pos[1] < paddle_y + 1:
            if self.paddle_pos <= self.ball_pos[0] <= self.paddle_pos + self.PADDLE_GRID_WIDTH:
                self.ball_pos[1] = paddle_y - 1
                self.ball_vel[1] *= -1
                
                # Add spin based on hit location
                offset = (self.ball_pos[0] - (self.paddle_pos + self.PADDLE_GRID_WIDTH / 2)) / (self.PADDLE_GRID_WIDTH / 2)
                self.ball_vel[0] += offset * 0.02
                # sfx: paddle_hit.wav

        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            bx, by = block['pos']
            if bx <= self.ball_pos[0] <= bx + 1 and by <= self.ball_pos[1] <= by + 1:
                # A simple collision response
                dx = self.ball_pos[0] - (bx + 0.5)
                dy = self.ball_pos[1] - (by + 0.5)
                if abs(dx) > abs(dy):
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                if block['type'] != 'indestructible':
                    self._create_particles(block['pos'])
                    reward += 1 if block['type'] == 'static' else 2
                    self.score += 10 if block['type'] == 'static' else 20
                    self.blocks_broken_total += 1
                    
                    # Increase ball speed every 10 blocks
                    if self.blocks_broken_total > 0 and self.blocks_broken_total % 10 == 0:
                        speed_norm = np.linalg.norm(self.ball_vel)
                        self.ball_vel = self.ball_vel / speed_norm * (speed_norm + self.BALL_SPEED_INCREMENT)

                    del self.blocks[i]
                    # sfx: block_break.wav
                else:
                    # sfx: block_hit_indestructible.wav
                    pass
                
                # Only handle one block collision per frame
                break
        
        # Normalize ball speed to prevent runaway acceleration from spin
        speed_norm = np.linalg.norm(self.ball_vel)
        if speed_norm > 0:
            max_speed = (self.BALL_START_SPEED + (self.blocks_broken_total // 10) * self.BALL_SPEED_INCREMENT) * 1.5
            current_speed = self.BALL_START_SPEED + (self.blocks_broken_total // 10) * self.BALL_SPEED_INCREMENT
            if speed_norm > max_speed:
                 self.ball_vel = self.ball_vel / speed_norm * current_speed

        return reward

    def _update_moving_blocks(self):
        for block in self.blocks:
            if block['type'] == 'moving':
                block['pos'][0] += block['move_dir'] * block['move_speed'] * 30
                if block['pos'][0] <= 0 or block['pos'][0] >= self.GRID_COLS - 1:
                    block['move_dir'] *= -1
                    block['pos'][0] = max(0, min(self.GRID_COLS - 1, block['pos'][0]))

    def _create_particles(self, grid_pos):
        screen_pos = self._iso_to_screen(grid_pos[0] + 0.5, grid_pos[1] + 0.5)
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(screen_pos),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'color': random.choice(self.COLOR_PARTICLE),
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                del self.particles[i]

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
            "stage": self.stage,
            "time_left": self.stage_timer,
        }

    def _render_game(self):
        # Render grid
        for r in range(self.GRID_ROWS + 1):
            start = self._iso_to_screen(0, r)
            end = self._iso_to_screen(self.GRID_COLS, r)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for c in range(self.GRID_COLS + 1):
            start = self._iso_to_screen(c, 0)
            end = self._iso_to_screen(c, self.GRID_ROWS)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Render blocks
        for block in self.blocks:
            self._render_iso_cube(block['pos'][0], block['pos'][1], block['color'])

        # Render paddle
        p1 = self._iso_to_screen(self.paddle_pos, self.GRID_ROWS - 1)
        p2 = self._iso_to_screen(self.paddle_pos + self.PADDLE_GRID_WIDTH, self.GRID_ROWS - 1)
        p3 = self._iso_to_screen(self.paddle_pos + self.PADDLE_GRID_WIDTH, self.GRID_ROWS)
        p4 = self._iso_to_screen(self.paddle_pos, self.GRID_ROWS)
        paddle_center = ((p1[0] + p3[0]) // 2, (p1[1] + p3[1]) // 2)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_PADDLE)
        pygame.gfxdraw.aacircle(self.screen, paddle_center[0], paddle_center[1], self.PADDLE_GRID_WIDTH * 6, self.COLOR_PADDLE_GLOW)


        # Render ball trail and ball
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, (*self.COLOR_BALL, alpha))
        
        ball_screen_pos = self._iso_to_screen(self.ball_pos[0], self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_iso_cube(self, x, y, color):
        """Renders a single isometric block."""
        top_points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1),
        ]
        
        # Darken side colors
        side_color1 = tuple(max(0, c - 40) for c in color)
        side_color2 = tuple(max(0, c - 60) for c in color)

        # Left side
        left_points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x, y + 1),
            (top_points[3][0], top_points[3][1] + self.ISO_TILE_H // 2),
            (top_points[0][0], top_points[0][1] + self.ISO_TILE_H // 2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, side_color2)
        
        # Right side
        right_points = [
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            (top_points[2][0], top_points[2][1] + self.ISO_TILE_H // 2),
            (top_points[1][0], top_points[1][1] + self.ISO_TILE_H // 2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, right_points, side_color1)
        
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)
    
    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT), (self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Balls
        balls_text = self.font_main.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - 100, 10))
        
        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=10)
        self.screen.blit(stage_text, stage_rect)

        # Timer
        time_text = self.font_main.render(f"TIME: {self.stage_timer // 30}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 + 100, 10))

        # Game Over / Win message
        if self.game_over:
            msg = self.font_large.render("GAME OVER", True, self.COLOR_BLOCK_STATIC)
            msg_rect = msg.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg, msg_rect)
        elif self.win:
            msg = self.font_large.render("YOU WIN!", True, self.COLOR_PADDLE)
            msg_rect = msg.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg, msg_rect)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game directly
    # To use, you might need to unset the dummy video driver:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # For human playback, we need a real display.
    # This is not available in the test environment.
    try:
        pygame.display.set_caption("Isometric Block Breaker")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    except pygame.error as e:
        print(f"Could not create display: {e}. Running headlessly.")
        screen = None

    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        # Event handling for human control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display if it exists
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset after a short delay
            if screen:
                pygame.time.wait(2000)
            obs, info = env.reset(seed=42)
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()