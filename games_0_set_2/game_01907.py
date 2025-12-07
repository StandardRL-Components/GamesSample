import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric arcade game where the player launches a ball to destroy all bricks
    within a time limit. The game prioritizes visual quality and satisfying game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑/↓ to aim, ←/→ to set power. Space to launch. Shift to reset aim."
    )

    # Short, user-facing description of the game
    game_description = (
        "Destroy all the bricks in a limited time by strategically launching a ball in an isometric arena."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 45
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_AIMER = (255, 80, 80)
    COLOR_BRICK_FACES = [(255, 90, 120), (255, 120, 90), (90, 150, 255), (120, 90, 255)]
    COLOR_BRICK_TOP = (240, 240, 250)
    COLOR_BRICK_SIDE = (180, 180, 200)

    # Isometric grid settings
    GRID_WIDTH, GRID_HEIGHT = 16, 16
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 100

    # Game mechanics
    NUM_BRICKS = 40
    BALL_MAX_HITS = 5
    BALL_LIFESPAN_SECONDS = 5
    BALL_SPEED_MIN, BALL_SPEED_MAX = 6, 14
    LAUNCH_ANGLE_MIN, LAUNCH_ANGLE_MAX = -45, 225
    LAUNCH_POWER_MIN, LAUNCH_POWER_MAX = 0.0, 1.0

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_timer = pygame.font.Font(None, 36)
        self.font_bricks = pygame.font.Font(None, 28)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.launch_angle = 0
        self.launch_power = 0
        self.ball = None
        self.bricks = []
        self.particles = []
        self.total_initial_bricks = 0
        self.last_space_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.launch_angle = 90.0  # Pointing straight up in iso grid
        self.launch_power = 0.5
        self.ball = None
        self.particles = []
        self.last_space_press = False

        self._generate_bricks()
        self.total_initial_bricks = len(self.bricks)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.timer -= 1

        # --- Handle player input ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        if self.ball is None: # Can only aim when ball is not in play
            # Angle adjustment
            if movement == 1: self.launch_angle += 2.5
            elif movement == 2: self.launch_angle -= 2.5
            self.launch_angle = np.clip(self.launch_angle, self.LAUNCH_ANGLE_MIN, self.LAUNCH_ANGLE_MAX)

            # Power adjustment
            if movement == 4: self.launch_power += 0.05
            elif movement == 3: self.launch_power -= 0.05
            self.launch_power = np.clip(self.launch_power, self.LAUNCH_POWER_MIN, self.LAUNCH_POWER_MAX)
            
            # Reset aim
            if shift_pressed:
                self.launch_angle = 90.0
                self.launch_power = 0.5

            # Launch ball (on key press, not hold)
            if space_pressed and not self.last_space_press:
                self._launch_ball()
        
        self.last_space_press = space_pressed

        # --- Update game state ---
        self._update_ball()
        reward += self._update_particles()
        
        # --- Check termination conditions ---
        if len(self.bricks) == 0 and not self.game_over:
            self.game_over = True
            reward += 100  # Victory bonus
        
        if self.timer <= 0 and not self.game_over:
            self.game_over = True
            reward -= 100 # Time out penalty

        terminated = self.game_over
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods ---

    def _generate_bricks(self):
        self.bricks = []
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        radius = 5
        
        available_positions = []
        for r in range(radius, 0, -1):
            for angle_deg in np.linspace(0, 360, int(1.5 * r * np.pi), endpoint=False):
                angle_rad = np.deg2rad(angle_deg)
                gx = int(center_x + r * np.cos(angle_rad))
                gy = int(center_y + r * np.sin(angle_rad))
                if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
                    if (gx, gy) not in [p['grid_pos'] for p in self.bricks]:
                         available_positions.append((gx, gy))

        # Ensure unique positions
        available_positions = list(set(available_positions))

        # Randomly sample from available positions
        num_to_place = min(self.NUM_BRICKS, len(available_positions))
        if num_to_place > 0:
            chosen_indices = self.np_random.choice(len(available_positions), num_to_place, replace=False)
            
            for i in chosen_indices:
                gx, gy = available_positions[i]
                sx, sy = self._iso_to_screen(gx, gy)
                color_index = self.np_random.integers(0, len(self.COLOR_BRICK_FACES))
                self.bricks.append({
                    'grid_pos': (gx, gy),
                    'screen_pos': (sx, sy),
                    'color': self.COLOR_BRICK_FACES[color_index],
                    'alive': True
                })

    def _launch_ball(self):
        angle_rad = math.radians(self.launch_angle)
        speed = self.BALL_SPEED_MIN + (self.BALL_SPEED_MAX - self.BALL_SPEED_MIN) * self.launch_power
        
        # Convert isometric angle to screen velocity
        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad)
        
        # Project this velocity into screen space
        screen_vx = (vx - vy) * 0.707 # Heuristic scaling
        screen_vy = (vx + vy) * 0.353 # Heuristic scaling

        self.ball = {
            'pos': np.array(self._iso_to_screen(self.GRID_WIDTH / 2, -2), dtype=np.float64),
            'vel': np.array([screen_vx, screen_vy], dtype=np.float64),
            'radius': 8,
            'hits_left': self.BALL_MAX_HITS,
            'life': self.BALL_LIFESPAN_SECONDS * self.FPS
        }

    def _update_ball(self):
        if self.ball is None:
            return

        self.ball['pos'] += self.ball['vel']
        self.ball['life'] -= 1

        # Ball timeout or out of hits
        if self.ball['life'] <= 0 or self.ball['hits_left'] <= 0:
            self.ball = None
            return

        # Wall collisions
        x, y = self.ball['pos']
        radius = self.ball['radius']
        if x < radius or x > self.SCREEN_WIDTH - radius:
            self.ball['vel'][0] *= -1
            self.ball['pos'][0] = np.clip(x, radius, self.SCREEN_WIDTH - radius)
        if y < radius or y > self.SCREEN_HEIGHT - radius:
            self.ball['vel'][1] *= -1
            self.ball['pos'][1] = np.clip(y, radius, self.SCREEN_HEIGHT - radius)

        # Brick collisions
        ball_rect = pygame.Rect(self.ball['pos'][0] - self.ball['radius'], self.ball['pos'][1] - self.ball['radius'], self.ball['radius']*2, self.ball['radius']*2)
        
        bricks_to_remove = []
        for i, brick in enumerate(self.bricks):
            brick_rect = pygame.Rect(brick['screen_pos'][0] - self.TILE_WIDTH_HALF, brick['screen_pos'][1] - self.TILE_HEIGHT_HALF, self.TILE_WIDTH_HALF*2, self.TILE_HEIGHT_HALF*2)
            if ball_rect.colliderect(brick_rect):
                bricks_to_remove.append(i)
                self.score += 1
                self.ball['hits_left'] -= 1
                self._create_particles(brick['screen_pos'], brick['color'])

                # Simple bounce logic
                dx = self.ball['pos'][0] - brick['screen_pos'][0]
                dy = self.ball['pos'][1] - brick['screen_pos'][1]
                if abs(dx) > abs(dy):
                    self.ball['vel'][0] *= -1
                else:
                    self.ball['vel'][1] *= -1
                
                # Prevent getting stuck
                self.ball['pos'] += self.ball['vel'] 

                # Only one brick hit per frame
                break 
        
        if bricks_to_remove:
            for i in sorted(bricks_to_remove, reverse=True):
                del self.bricks[i]

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        reward_from_particles = 0
        
        num_bricks_before = len(self.bricks)
        
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        num_bricks_after = len(self.bricks)
        
        # This is a proxy for the reward from destroying a brick in the previous frame
        # We check the change in brick count before and after the particle update
        # which is a bit indirect. A better way is to reward directly in _update_ball
        # but let's stick to the original logic as much as possible.
        # The original code's reward logic for particles was flawed, let's fix it slightly.
        # The reward should be tied to destroying bricks, not particle updates.
        # The step function already adds to score in _update_ball, so we'll use that.
        # Let's check `_update_ball` again. `self.score` is incremented, but not returned as reward.
        # The original `_update_particles` logic was:
        # num_bricks_before = len(self.bricks)
        # ...
        # num_bricks_after = len(self.bricks)
        # if num_bricks_before > num_bricks_after:
        #     reward_from_particles = 1.0 * (num_bricks_before - num_bricks_after)
        # This is flawed because `_update_particles` doesn't remove bricks. `_update_ball` does.
        # A simple fix is to return the number of bricks destroyed in the current step.
        bricks_destroyed_this_step = self.total_initial_bricks - len(self.bricks) - self.score
        reward_from_particles = bricks_destroyed_this_step

        return reward_from_particles


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
            "timer": self.timer / self.FPS,
            "bricks_left": len(self.bricks),
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, pos, color, height=8):
        x, y = pos
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        top_points = [
            (x, y - h),
            (x + w, y),
            (x, y + h),
            (x - w, y),
        ]
        
        side_color = self.COLOR_BRICK_SIDE
        
        # Left side
        pygame.gfxdraw.filled_polygon(surface, [(x - w, y), (x, y + h), (x, y + h + height), (x - w, y + height)], side_color)
        pygame.gfxdraw.aapolygon(surface, [(x - w, y), (x, y + h), (x, y + h + height), (x - w, y + height)], side_color)
        
        # Right side
        pygame.gfxdraw.filled_polygon(surface, [(x + w, y), (x, y + h), (x, y + h + height), (x + w, y + height)], side_color)
        pygame.gfxdraw.aapolygon(surface, [(x + w, y), (x, y + h), (x, y + h + height), (x + w, y + height)], side_color)
        
        # Top face
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw launch platform
        platform_pos = self._iso_to_screen(self.GRID_WIDTH / 2, -2)
        self._draw_iso_cube(self.screen, platform_pos, (100, 100, 120), height=4)

        # Sort bricks and ball for correct Z-ordering
        render_queue = []
        for brick in self.bricks:
            render_queue.append(('brick', brick['screen_pos'], brick))
        if self.ball:
            render_queue.append(('ball', self.ball['pos'], self.ball))
        
        render_queue.sort(key=lambda item: item[1][1])

        # Render sorted items
        for item_type, pos, data in render_queue:
            if item_type == 'brick':
                self._draw_iso_cube(self.screen, data['screen_pos'], data['color'])
            elif item_type == 'ball':
                # Glow effect
                glow_radius = int(data['radius'] * 1.8)
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surface, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
                
                # Ball
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), data['radius'], self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), data['radius'], self.COLOR_PLAYER)

        # Render particles (on top)
        for p in self.particles:
            size = max(0, p['radius'] * (p['life'] / 30.0))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

        # Render aiming indicator if no ball
        if self.ball is None:
            start_pos = self._iso_to_screen(self.GRID_WIDTH / 2, -2)
            angle_rad = math.radians(self.launch_angle)
            length = 30 + 100 * self.launch_power
            
            vx = length * math.cos(angle_rad)
            vy = length * math.sin(angle_rad)
            screen_vx = (vx - vy) * 0.707
            screen_vy = (vx + vy) * 0.353
            end_pos = (start_pos[0] + screen_vx, start_pos[1] + screen_vy)
            
            pygame.draw.line(self.screen, self.COLOR_AIMER, start_pos, end_pos, 2)
            pygame.draw.circle(self.screen, self.COLOR_AIMER, end_pos, 4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Bricks left
        bricks_left = len(self.bricks)
        bricks_text = self.font_bricks.render(f"BRICKS: {bricks_left}", True, self.COLOR_UI_TEXT)
        bricks_rect = bricks_text.get_rect(midbottom=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(bricks_text, bricks_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    # and visualization.
    env = GameEnv()
    
    # Setup a display for human playing
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    pygame.display.set_caption("Isometric Brick Breaker")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to our action space
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }

    while not done:
        # Default action is NO-OP
        action = [0, 0, 0]

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Movement
        for key, (move, _, _) in key_map.items():
            if keys[key]:
                action[0] = move
                break
        
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing or resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()