import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Ensure Pygame runs headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro-arcade block-breaking game.
    The player controls a pivoting paddle to bounce a ball and destroy a grid of blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to rotate the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a pivoting paddle to smash blocks with a bouncing ball in a vibrant, retro-arcade world."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30 # For auto_advance=True

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (40, 0, 70)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_PADDLE_GLOW = (0, 150, 255, 50)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 0, 100)
    COLOR_WALLS = (100, 100, 150, 150)
    BLOCK_COLORS = [
        (255, 0, 128), (255, 128, 0), (0, 200, 200), (128, 0, 255), (0, 255, 0)
    ]
    COLOR_TEXT = (255, 255, 255)

    # Paddle properties
    PADDLE_PIVOT = pygame.Vector2(SCREEN_WIDTH // 2, 380)
    PADDLE_LENGTH = 100
    PADDLE_THICKNESS = 6
    PADDLE_ROTATION_SPEED = 2.0  # degrees per step
    PADDLE_MIN_ANGLE = -80
    PADDLE_MAX_ANGLE = 80

    # Ball properties
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    BALL_SPEED_INCREMENT = 0.5

    # Block properties
    BLOCK_ROWS = 5
    BLOCK_COLS = 8
    BLOCK_WIDTH = 60
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 10
    BLOCK_START_Y = 50

    # Game properties
    MAX_LIVES = 3
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.lives = 0
        self.paddle_angle = 0.0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_on_paddle = True
        self.ball_speed = 0.0
        self.blocks = []
        self.particles = []
        self.blocks_destroyed_total = 0
        self.chain_hits_this_turn = 0
        self.np_random = None

        # Reward tracking flags
        self.reward_info = {}

        # The original code called reset() here, which is not standard.
        # We will initialize the RNG here and let the user call reset().
        self.np_random = np.random.default_rng()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.lives = self.MAX_LIVES
        self.paddle_angle = 0.0
        self.ball_on_paddle = True
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()
        self._create_blocks()
        self.particles = []
        self.blocks_destroyed_total = 0
        self.chain_hits_this_turn = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.reward_info = {
            "block_hit": False,
            "paddle_hit": False,
            "chain_bonus": False,
            "safe_play_penalty": False,
            "lost_life": False
        }

        terminated = False
        truncated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()

        reward = self._calculate_reward()
        self.score += reward
        self.steps += 1
        
        if self.game_over:
            terminated = True
        if self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_pressed = action[1] == 1
        
        # Rotate paddle
        if movement == 3:  # Left
            self.paddle_angle -= self.PADDLE_ROTATION_SPEED
        elif movement == 4:  # Right
            self.paddle_angle += self.PADDLE_ROTATION_SPEED
        
        self.paddle_angle = np.clip(self.paddle_angle, self.PADDLE_MIN_ANGLE, self.PADDLE_MAX_ANGLE)

        # Launch ball
        if space_pressed and self.ball_on_paddle:
            self.ball_on_paddle = False
            launch_angle_rad = math.radians(self.paddle_angle - 90)
            self.ball_vel = pygame.Vector2(math.cos(launch_angle_rad), math.sin(launch_angle_rad)) * self.ball_speed
            # sfx: launch_ball

    def _update_game_state(self):
        if self.ball_on_paddle:
            self._reset_ball()
        else:
            self._update_ball()
        
        self._update_particles()
        
        if not self.blocks:
            self.game_won = True
            self.game_over = True
            
        if self.lives <= 0:
            self.game_over = True

    def _update_ball(self):
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = self.BALL_RADIUS
            # sfx: wall_bounce
        if self.ball_pos.x >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = self.SCREEN_WIDTH - self.BALL_RADIUS
            # sfx: wall_bounce
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        paddle_p1, paddle_p2 = self._get_paddle_endpoints()
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        if self.ball_vel.y > 0 and ball_rect.clipline(paddle_p1, paddle_p2):
            paddle_normal = pygame.Vector2(math.sin(math.radians(self.paddle_angle)), -math.cos(math.radians(self.paddle_angle)))
            self.ball_vel.reflect_ip(paddle_normal)
            self.ball_pos.y = self.PADDLE_PIVOT.y - self.PADDLE_THICKNESS - self.BALL_RADIUS
            self.reward_info["paddle_hit"] = True
            self.chain_hits_this_turn = 0
            # sfx: paddle_hit
            
            if abs(self.paddle_angle) < 10:
                self.reward_info["safe_play_penalty"] = True

        # Block collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        block_rects = [block['rect'] for block in self.blocks]
        hit_block_idx = ball_rect.collidelist(block_rects)
        if hit_block_idx != -1:
            hit_block_info = self.blocks.pop(hit_block_idx)
            hit_block = hit_block_info['rect']
            
            dx = self.ball_pos.x - hit_block.centerx
            dy = self.ball_pos.y - hit_block.centery
            w = (self.BALL_RADIUS + hit_block.width / 2)
            h = (self.BALL_RADIUS + hit_block.height / 2)
            
            if abs(dx) / w > abs(dy) / h:
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1
                
            self.reward_info["block_hit"] = True
            self.chain_hits_this_turn += 1
            if self.chain_hits_this_turn > 1:
                self.reward_info["chain_bonus"] = True
            
            self.blocks_destroyed_total += 1
            self._create_particles(hit_block.center, self.BLOCK_COLORS[hit_block_info['row_idx'] % len(self.BLOCK_COLORS)])
            # sfx: block_destroy
            
            if self.blocks_destroyed_total > 0 and self.blocks_destroyed_total % 10 == 0:
                self.ball_speed += self.BALL_SPEED_INCREMENT
                if self.ball_vel.length() > 0:
                    self.ball_vel = self.ball_vel.normalize() * self.ball_speed

        # Lose life
        if self.ball_pos.y > self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            self.ball_on_paddle = True
            self.reward_info["lost_life"] = True
            # sfx: lose_life

    def _calculate_reward(self):
        reward = -0.02

        if self.reward_info["paddle_hit"]:
            reward += 0.1
        if self.reward_info["safe_play_penalty"]:
            reward -= 20.0
        if self.reward_info["block_hit"]:
            reward += 1.0
        if self.reward_info["chain_bonus"]:
            reward += 5.0
        
        if self.game_over:
            if self.game_won:
                reward += 100.0
            elif self.lives <= 0:
                reward -= 100.0

        return reward

    def _get_observation(self):
        self._render_background()
        self._render_walls()
        self._render_particles()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks)
        }

    # --- Rendering Methods ---
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_walls(self):
        wall_thickness = int(self.BALL_RADIUS * 1.5)
        wall_rects = [
            pygame.Rect(0, 0, self.SCREEN_WIDTH, wall_thickness),
            pygame.Rect(0, 0, wall_thickness, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - wall_thickness, 0, wall_thickness, self.SCREEN_HEIGHT),
        ]
        for rect in wall_rects:
             pygame.draw.rect(self.screen, self.COLOR_WALLS, rect)


    def _render_paddle(self):
        p1, p2 = self._get_paddle_endpoints()
        pygame.draw.line(self.screen, self.COLOR_PADDLE_GLOW, p1, p2, self.PADDLE_THICKNESS + 8)
        pygame.draw.line(self.screen, self.COLOR_PADDLE, p1, p2, self.PADDLE_THICKNESS)

    def _render_ball(self):
        if self.game_over and not self.game_won:
            return
            
        pos = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_blocks(self):
        for block_info in self.blocks:
            block_rect = block_info['rect']
            color = self.BLOCK_COLORS[block_info['row_idx'] % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c * 0.7) for c in color), block_rect, 2, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            radius = max(0, int(p['radius']))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, color)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 20, 20))
        
        if self.game_over:
            if self.game_won:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_BALL)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.BLOCK_COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Helper Methods ---
    def _create_blocks(self):
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_START_Y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_info = {
                    'rect': pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    'row_idx': i
                }
                self.blocks.append(block_info)

    def _reset_ball(self):
        angle_rad = math.radians(self.paddle_angle - 90)
        offset_len = (self.PADDLE_LENGTH / 4) + self.BALL_RADIUS
        offset = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * offset_len
        self.ball_pos = self.PADDLE_PIVOT + offset
        self.ball_vel = pygame.Vector2(0, 0)

    def _get_paddle_endpoints(self):
        half_len_vec = pygame.Vector2(self.PADDLE_LENGTH / 2, 0).rotate(-self.paddle_angle)
        p1 = self.PADDLE_PIVOT - half_len_vec
        p2 = self.PADDLE_PIVOT + half_len_vec
        return p1, p2

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': int(self.np_random.integers(15, 31)),
                'max_life': 30,
                'color': color
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] > 0 and p['radius'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def close(self):
        pygame.quit()

# This is for testing purposes if the file is run directly
if __name__ == '__main__':
    # To make it playable, we need a screen to display the frames
    # Un-comment the next two lines to run locally
    # os.environ.pop("SDL_VIDEODRIVER", None)
    # real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Smashout - Human Player")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a display if not running headlessly for testing
    try:
        real_screen = pygame.display.get_surface()
        if real_screen is None:
            raise pygame.error
        is_playable = True
    except pygame.error:
        is_playable = False
        print("Running in headless mode. No display will be shown.")

    while not done:
        action = [0, 0, 0] # Default no-op action
        if is_playable:
            # Map keyboard keys to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            
            movement = 0 # No-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0 # Not used
            
            action = [movement, space_held, shift_held]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if is_playable:
            # Render the observation from the environment to the real screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()

        if not is_playable and env.steps % 100 == 0:
            print(f"Step: {env.steps}, Score: {info['score']:.2f}, Blocks: {info['blocks_left']}")


    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()