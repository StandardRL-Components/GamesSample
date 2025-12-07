import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:58:04.140971
# Source Brief: brief_01401.md
# Brief Index: 1401
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Neon Rhythm Racer: A Gymnasium environment where an agent races through a
    vibrant neon cityscape, hitting rhythm checkpoints to boost speed and
    activating portals to navigate dynamic tracks.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "A fast-paced rhythm racer. Steer through a neon cityscape, hit checkpoints to the beat, "
        "and use portals to teleport between lanes."
    )
    user_guide = (
        "Controls: ←→ to change lanes. Press Shift to hit checkpoints and Space to use portals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 0, 25)
    COLOR_GRID = (40, 20, 80)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_CHECKPOINT = (50, 255, 50)
    COLOR_CHECKPOINT_GLOW = (25, 150, 25)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (150, 25, 25)
    COLOR_PORTAL = (255, 0, 255)
    COLOR_PORTAL_GLOW = (150, 0, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)

    # Game parameters
    LANE_COUNT = 3
    LANE_WIDTH = 120
    MAX_EPISODE_STEPS = 2000
    PLAYER_Y_POS = SCREEN_HEIGHT * 0.8
    CHECKPOINT_HIT_WINDOW = 40 # pixels

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        # This is called in reset(), so we don't need it here, but it's good practice
        # to have the attributes defined in __init__
        self._initialize_state()
        
        # Run validation check
        # self.validate_implementation() # This is better run outside the class by a test suite

    def _initialize_state(self):
        """Initializes all game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_lane = 1 # 0, 1, 2
        self.player_visual_x = self.SCREEN_WIDTH / 2
        self.player_target_x = self.SCREEN_WIDTH / 2
        
        # Game progression state
        self.world_scroll_y = 0.0
        self.speed = 5.0
        self.base_speed = 5.0
        self.combo = 1
        self.max_combo_this_run = 1
        
        # Entity lists
        self.checkpoints = []
        self.obstacles = []
        self.portals = []
        self.particles = []
        self.background_stars = []

        # Action state trackers
        self.prev_shift_held = False
        self.prev_space_held = False
        self.current_shift_press = False
        self.current_space_press = False
        
        # Track generation
        self.next_segment_y = 0
        self._generate_initial_track()
        self._generate_stars(200)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for existing to encourage speed

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self._handle_input(movement, space_held, shift_held)
        self._update_player_position()
        self._update_world_scroll()
        
        reward += self._update_checkpoints()
        self._update_obstacles()
        self._update_portals()
        self._update_particles()

        self._cull_offscreen_entities()
        self._generate_track_segments()

        # Check for termination conditions
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
        
        terminated = self.game_over
        if terminated and self.combo > 0: # Only penalize for crashing, not for time-out
             reward -= 50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is not used in this environment
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement: 3=left, 4=right
        if movement == 3:
            self.player_lane = max(0, self.player_lane - 1)
        elif movement == 4:
            self.player_lane = min(self.LANE_COUNT - 1, self.player_lane + 1)
        
        lane_center_x = (self.SCREEN_WIDTH / 2) + (self.player_lane - 1) * self.LANE_WIDTH
        self.player_target_x = lane_center_x
        
        self.current_space_press = space_held and not self.prev_space_held
        self.current_shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player_position(self):
        # Smooth interpolation for player movement
        self.player_visual_x += (self.player_target_x - self.player_visual_x) * 0.25

    def _update_world_scroll(self):
        # Speed is boosted by combo
        self.speed = self.base_speed + (self.combo - 1) * 0.5
        self.world_scroll_y += self.speed

    def _update_checkpoints(self):
        reward = 0
        for checkpoint in self.checkpoints:
            checkpoint['y'] += self.speed
            
            # Check for successful hit
            if self.current_shift_press and checkpoint['lane'] == self.player_lane:
                if abs(checkpoint['y'] - self.PLAYER_Y_POS) < self.CHECKPOINT_HIT_WINDOW:
                    if not checkpoint['hit']:
                        checkpoint['hit'] = True
                        self.combo += 1
                        self.max_combo_this_run = max(self.combo, self.max_combo_this_run)
                        self.score += 10 * (self.combo // 5)
                        reward += 5.0 # Big reward for hitting
                        # // SFX: Checkpoint hit
                        self._create_particles(self.player_visual_x, self.PLAYER_Y_POS, self.COLOR_CHECKPOINT, 30)

            # Check for missed checkpoint
            if not checkpoint['hit'] and checkpoint['y'] > self.PLAYER_Y_POS + self.CHECKPOINT_HIT_WINDOW:
                if self.combo > 0:
                    reward -= 10.0 # Big penalty for miss
                    # // SFX: Missed checkpoint
                self.combo = 0 # Reset combo
                self.game_over = True # End run on miss
                checkpoint['hit'] = True # Mark as processed
        return reward

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle['y'] += self.speed
            # Collision check
            player_rect = pygame.Rect(self.player_visual_x - 15, self.PLAYER_Y_POS - 15, 30, 30)
            obstacle_rect = pygame.Rect(obstacle['x'] - obstacle['w']/2, obstacle['y'] - obstacle['h']/2, obstacle['w'], obstacle['h'])
            if player_rect.colliderect(obstacle_rect):
                self.game_over = True
                self.combo = 0
                # // SFX: Crash
                self._create_particles(self.player_visual_x, self.PLAYER_Y_POS, self.COLOR_OBSTACLE, 50)

    def _update_portals(self):
        for portal in self.portals:
            portal['y'] += self.speed
            if self.current_space_press and portal['lane_from'] == self.player_lane:
                if abs(portal['y'] - self.PLAYER_Y_POS) < self.CHECKPOINT_HIT_WINDOW:
                    self.player_lane = portal['lane_to']
                    self.player_target_x = (self.SCREEN_WIDTH / 2) + (self.player_lane - 1) * self.LANE_WIDTH
                    portal['y'] = self.SCREEN_HEIGHT * 2 # Move offscreen after use
                    # // SFX: Portal warp
                    self._create_particles(self.player_visual_x, self.PLAYER_Y_POS, self.COLOR_PORTAL, 20)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _cull_offscreen_entities(self):
        cull_y = self.SCREEN_HEIGHT + 50
        self.checkpoints = [c for c in self.checkpoints if c['y'] < cull_y]
        self.obstacles = [o for o in self.obstacles if o['y'] < cull_y]
        self.portals = [p for p in self.portals if p['y'] < cull_y]

    def _get_lane_x(self, lane_idx):
        return (self.SCREEN_WIDTH / 2) + (lane_idx - 1) * self.LANE_WIDTH

    def _generate_initial_track(self):
        for i in range(10):
            self._generate_track_segment()

    def _generate_track_segments(self):
        while self.next_segment_y < self.world_scroll_y + self.SCREEN_HEIGHT:
            self._generate_track_segment()

    def _generate_track_segment(self):
        y_pos = -self.next_segment_y + self.world_scroll_y
        
        # Add a checkpoint
        lane = self.np_random.integers(0, self.LANE_COUNT)
        self.checkpoints.append({'lane': lane, 'y': y_pos, 'hit': False})

        # Occasionally add an obstacle or a portal
        if self.np_random.random() < 0.3 + min(0.3, self.steps / 5000): # Increase chance over time
            obstacle_lane = self.np_random.integers(0, self.LANE_COUNT)
            if obstacle_lane != lane: # Don't block the checkpoint
                self.obstacles.append({
                    'x': self._get_lane_x(obstacle_lane), 
                    'y': y_pos - 100, 
                    'w': 50, 'h': 20
                })
        elif self.np_random.random() < 0.1:
            from_lane = self.np_random.integers(0, self.LANE_COUNT)
            to_lane_offset = self.np_random.choice([-1, 1])
            to_lane = (from_lane + to_lane_offset)
            # Ensure to_lane is within bounds [0, LANE_COUNT-1]
            if not (0 <= to_lane < self.LANE_COUNT):
                to_lane = (from_lane - to_lane_offset)

            self.portals.append({
                'lane_from': from_lane, 
                'lane_to': to_lane, 
                'y': y_pos - 150
            })

        self.next_segment_y += 250 # Distance between segments
    
    def _generate_stars(self, count):
        for _ in range(count):
            self.background_stars.append({
                'x': self.np_random.uniform(0, self.SCREEN_WIDTH),
                'y': self.np_random.uniform(0, self.SCREEN_HEIGHT),
                'depth': self.np_random.uniform(0.1, 0.6) # For parallax
            })

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render background stars (parallax)
        for star in self.background_stars:
            star['y'] += self.speed * star['depth']
            if star['y'] > self.SCREEN_HEIGHT:
                star['y'] = 0
                star['x'] = self.np_random.uniform(0, self.SCREEN_WIDTH)
            size = int(star['depth'] * 3)
            pygame.draw.rect(self.screen, (255,255,255), (int(star['x']), int(star['y']), size, size))

        # Render track grid
        for i in range(-1, 2):
            x = self.SCREEN_WIDTH / 2 + i * self.LANE_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 2)
        
        # Render entities
        for portal in self.portals:
            self._draw_glowing_rect(self._get_lane_x(portal['lane_from']), portal['y'], 80, 20, self.COLOR_PORTAL, self.COLOR_PORTAL_GLOW)
        for obstacle in self.obstacles:
            self._draw_glowing_rect(obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h'], self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        for checkpoint in self.checkpoints:
            if not checkpoint['hit']:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
                self._draw_glowing_rect(self._get_lane_x(checkpoint['lane']), checkpoint['y'], 60 + pulse, 15 + pulse/2, self.COLOR_CHECKPOINT, self.COLOR_CHECKPOINT_GLOW)
        
        # Render player
        self._draw_player()
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = p['color'] + (alpha,)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p['pos'][0]-1), int(p['pos'][1]-1)))
            
    def _draw_player(self):
        x, y = int(self.player_visual_x), int(self.PLAYER_Y_POS)
        points = [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)]
        glow_points = [(x, y - 24), (x - 18, y + 12), (x + 18, y + 12)]
        
        # Glow
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        # Player
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_glowing_rect(self, x, y, w, h, color, glow_color):
        rect = pygame.Rect(x - w/2, y - h/2, w, h)
        glow_rect = pygame.Rect(x - w/2 - 5, y - h/2 - 5, w + 10, h + 10)
        
        # Simple glow effect by drawing a larger, semi-transparent rect behind
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, glow_color + (100,), shape_surf.get_rect(), border_radius=5)
        self.screen.blit(shape_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_ui(self):
        # Combo Multiplier
        combo_text = f"COMBO: {self.combo}x"
        self._draw_text(combo_text, self.font_ui, (20, 20), "topleft")

        # Max Combo
        max_combo_text = f"MAX: {self.max_combo_this_run}x"
        self._draw_text(max_combo_text, self.font_ui, (20, 50), "topleft")
        
        # Speed
        speed_text = f"SPEED: {self.speed:.1f}"
        self._draw_text(speed_text, self.font_ui, (self.SCREEN_WIDTH - 20, 20), "topright")

        # Timer/Steps
        time_text = f"{self.steps:04d}"
        self._draw_text(time_text, self.font_timer, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20), "midbottom")

    def _draw_text(self, text, font, pos, align="center"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        
        setattr(text_rect, align, pos)
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "speed": self.speed
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Rhythm Racer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # Arrow Keys: Left/Right
    # Spacebar: Activate Portal
    # Left Shift: Hit Checkpoint
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Reset for a new game after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        clock.tick(30) # Run at 30 FPS
        
    env.close()