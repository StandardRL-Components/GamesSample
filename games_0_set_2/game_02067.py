import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to move the cursor. Hold space to draw a track. Press shift to undo the last segment."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a path for your sled to guide it through challenging stages to the finish line. Be quick and efficient to maximize your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = pygame.Color("#1E1E2E")
        self.COLOR_GRID = pygame.Color("#313244")
        self.COLOR_TERRAIN = pygame.Color("#45475A")
        self.COLOR_TRACK = pygame.Color("#89B4FA")
        self.COLOR_SLED = pygame.Color("#F38BA8")
        self.COLOR_CURSOR = pygame.Color("#A6E3A1")
        self.COLOR_PARTICLE = pygame.Color("#CDD6F4")
        self.COLOR_START = pygame.Color("#A6E3A1")
        self.COLOR_FINISH = pygame.Color("#F38BA8")
        self.COLOR_TEXT = pygame.Color("#CDD6F4")
        
        # Game constants
        self.GRAVITY = 0.3
        self.FRICTION = 0.99
        self.CURSOR_SPEED = 8
        self.MAX_STEPS = 5000
        self.MIN_SEGMENT_LENGTH = 10

        # Game state variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.total_score = 0
        self.stage = 1
        self.game_over = False
        self.win = False
        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.track_points = []
        self.terrain_points = []
        self.stage_finish_x = 0
        self.particles = []
        self.last_shift_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.total_score = 0
        self.stage = 1
        self.game_over = False
        self.win = False
        self.particles = []
        self.last_shift_press = False
        
        self._setup_stage(self.stage)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        self.track_points = []
        self.terrain_points = []
        
        if stage_num == 1:
            start_pos = pygame.math.Vector2(50, 100)
            self.stage_finish_x = 600
            self.terrain_points = [start_pos, pygame.math.Vector2(self.stage_finish_x, 350)]
        elif stage_num == 2:
            start_pos = pygame.math.Vector2(50, 150)
            self.stage_finish_x = 620
            self.terrain_points = [
                start_pos,
                pygame.math.Vector2(200, 250),
                pygame.math.Vector2(350, 100),
                pygame.math.Vector2(500, 200),
                pygame.math.Vector2(self.stage_finish_x, 150)
            ]
        elif stage_num == 3:
            start_pos = pygame.math.Vector2(50, 50)
            self.stage_finish_x = 620
            self.terrain_points = [
                start_pos, pygame.math.Vector2(250, 150),
                # Gap starts here
                pygame.math.Vector2(400, 150), pygame.math.Vector2(self.stage_finish_x, 300)
            ]

        self.sled_pos = start_pos.copy()
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.cursor_pos = start_pos + pygame.math.Vector2(50, 0)
        self.track_points.append(start_pos.copy())
        
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        reward += self._handle_input(movement, space_held, shift_held)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        
        # Base reward for survival
        reward -= 0.01

        # Reward for sled speed
        reward += max(0, self.sled_vel.x * 0.01)
        if self.sled_vel.length() < 0.5:
            reward -= 0.02

        terminated, event_reward = self._check_termination_and_progress()
        reward += event_reward
        
        self.total_score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Move Cursor ---
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED  # Right
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # --- Draw Track ---
        if space_held:
            last_point = self.track_points[-1]
            if self.cursor_pos.distance_to(last_point) > self.MIN_SEGMENT_LENGTH:
                # # SFX: Draw line
                self.track_points.append(self.cursor_pos.copy())
                
                # Penalize overly steep upward slopes that kill momentum
                new_segment = self.cursor_pos - last_point
                if new_segment.y < 0 and new_segment.length() > 0:
                    angle_rad = math.atan2(-new_segment.y, new_segment.x)
                    if math.degrees(angle_rad) > 45:
                        reward -= 0.1 # Risky angle penalty
        
        # --- Undo Track ---
        if shift_held and not self.last_shift_press:
            if len(self.track_points) > 1:
                # # SFX: Undo
                self.track_points.pop()
        self.last_shift_press = shift_held
        
        return reward

    def _update_physics(self):
        # Apply gravity
        self.sled_vel.y += self.GRAVITY
        
        # Tentative new position
        next_pos = self.sled_pos + self.sled_vel

        # Find ground under sled
        ground_y, ground_segment = self._find_ground_beneath(next_pos.x)

        if ground_segment and next_pos.y >= ground_y - 5: # Collision threshold
            # # SFX: Sled scraping on track
            p1, p2 = ground_segment
            
            # Snap position to the ground
            self.sled_pos.x = next_pos.x
            self.sled_pos.y = ground_y
            
            # Re-calculate velocity based on slope
            segment_vec = p2 - p1
            if segment_vec.length() > 0:
                # Project velocity onto the track direction
                seg_dir = segment_vec.normalize()
                dot_product = self.sled_vel.dot(seg_dir)
                self.sled_vel = seg_dir * dot_product

                # Apply friction
                self.sled_vel *= self.FRICTION
            else:
                self.sled_vel = pygame.math.Vector2(0, 0)
            
            # Add particles for sliding
            if self.sled_vel.length() > 1:
                self._create_particles(self.sled_pos, 5)

        else: # In the air
            self.sled_pos = next_pos

    def _find_ground_beneath(self, x_pos):
        highest_y = float('inf')
        ground_segment = None

        all_tracks = [self.terrain_points]
        if len(self.track_points) > 1:
            all_tracks.append(self.track_points)

        for track in all_tracks:
            for i in range(len(track) - 1):
                p1 = track[i]
                p2 = track[i+1]
                
                # Ensure x_pos is within the segment's horizontal bounds
                if (p1.x <= x_pos <= p2.x) or (p2.x <= x_pos <= p1.x):
                    # Avoid division by zero for vertical lines
                    if abs(p2.x - p1.x) < 1e-6:
                        continue
                    
                    # Linear interpolation to find y on the segment at x_pos
                    y_on_segment = p1.y + (p2.y - p1.y) * ((x_pos - p1.x) / (p2.x - p1.x))
                    
                    if y_on_segment < highest_y:
                        highest_y = y_on_segment
                        ground_segment = (p1, p2)

        return (highest_y, ground_segment) if ground_segment else (float('inf'), None)

    def _check_termination_and_progress(self):
        # --- Check Termination ---
        if not (0 < self.sled_pos.y < self.HEIGHT) or not (0 < self.sled_pos.x < self.WIDTH):
            # # SFX: Crash
            self.game_over = True
            return True, -100 # Heavy penalty for crashing

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, -50 # Penalty for timeout

        # --- Check Progress ---
        if self.sled_pos.x > self.stage_finish_x:
            # # SFX: Stage complete
            self.stage += 1
            if self.stage > 3:
                # # SFX: Win game
                self.win = True
                return True, 50 # Big reward for winning
            else:
                self._setup_stage(self.stage)
                return False, 10 # Reward for finishing a stage
        
        return False, 0

    def _create_particles(self, pos, count):
        if self.np_random is None: self.reset()
        for _ in range(count):
            particle_pos = pos.copy() + pygame.math.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-2, 2))
            particle_vel = pygame.math.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1, -0.5))
            lifetime = self.np_random.integers(10, 25)
            radius = self.np_random.uniform(1, 3)
            self.particles.append({'pos': particle_pos, 'vel': particle_vel, 'life': lifetime, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        # --- Render Game ---
        self.screen.fill(self.COLOR_BG)
        
        # Grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Terrain
        if len(self.terrain_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TERRAIN, False, [(int(p.x), int(p.y)) for p in self.terrain_points], 5)

        # Player-drawn track
        if len(self.track_points) > 1:
            points = [(int(p.x), int(p.y)) for p in self.track_points]
            # FIX: Use pygame.draw.aalines instead of pygame.gfxdraw.aalines
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, points, 3)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['pos'].x), int(p['pos'].y)), max(0, int(p['radius'])))
            
        # Sled
        sled_rect = pygame.Rect(0, 0, 20, 10)
        sled_rect.center = (int(self.sled_pos.x), int(self.sled_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect, border_radius=3)
        
        # Start/Finish Lines
        if self.terrain_points:
            start_pos = self.terrain_points[0]
            pygame.draw.line(self.screen, self.COLOR_START, (start_pos.x, 0), (start_pos.x, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.stage_finish_x, 0), (self.stage_finish_x, self.HEIGHT), 2)
        
        # Cursor
        if not self.game_over and not self.win:
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 10, self.COLOR_CURSOR)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x-5, self.cursor_pos.y), (self.cursor_pos.x+5, self.cursor_pos.y))
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x, self.cursor_pos.y-5), (self.cursor_pos.x, self.cursor_pos.y+5))

        # --- Render UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"Score: {int(self.total_score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        stage_surf = self.font_ui.render(f"Stage: {self.stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.WIDTH - stage_surf.get_width() - 10, 10))

        if self.game_over:
            msg_surf = self.font_msg.render("GAME OVER", True, self.COLOR_FINISH)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)
        elif self.win:
            msg_surf = self.font_msg.render("YOU WIN!", True, self.COLOR_START)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)
            
    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "stage": self.stage,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To play the game manually
    # Make sure to remove the dummy video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Drawer")

    # Game loop
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Render one last time to show the game over message
            draw_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_surf.blit(draw_surf, (0, 0))
            pygame.display.flip()
            # Optional: auto-reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the display window
        draw_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surf.blit(draw_surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)

    env.close()