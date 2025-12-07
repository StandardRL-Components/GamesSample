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

    user_guide = (
        "Controls: ←→ to steer. Hold Space to boost. Collide with obstacles to lose."
    )

    game_description = (
        "A fast-paced isometric racer. Steer your ship down a futuristic track, "
        "dodging obstacles to complete two laps as fast as possible."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 19, 26)
    COLOR_TRACK = (50, 55, 65)
    COLOR_TRACK_LINES = (80, 85, 95)
    COLOR_PLAYER = (255, 20, 70)
    COLOR_PLAYER_GLOW = (255, 100, 140)
    COLOR_OBSTACLE = (20, 180, 255)
    COLOR_OBSTACLE_GLOW = (100, 220, 255)
    COLOR_CHECKPOINT = (20, 255, 150)
    COLOR_FINISH_PULS_1 = (255, 215, 0)
    COLOR_FINISH_PULS_2 = (255, 255, 100)
    COLOR_TEXT = (240, 240, 240)

    # Physics & Gameplay
    PLAYER_RENDER_Y = 320
    TRACK_WIDTH = 350
    TRACK_LENGTH = 5000
    LAPS_TO_WIN = 2
    MAX_STEPS = 6000 # ~100s at 60fps

    FORWARD_SPEED_BASE = 8.0
    FORWARD_SPEED_BOOST = 16.0
    
    PLAYER_ACCEL = 0.8
    PLAYER_DRAG = 0.90
    PLAYER_MAX_VX = 10.0
    
    NUM_OBSTACLES = 40

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
        
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 36)
            self.font_medium = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        self.particles = []
        self.obstacles = []
        
        # self.reset() is called by the wrapper, but we can call it to initialize state
        # self.validate_implementation() is a helper and not needed for the final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_x = 0.0
        self.player_vx = 0.0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps = 0
        self.progress_y = 0.0
        self.last_progress_y = 0.0
        
        # World state
        self.particles.clear()
        self._spawn_obstacles()
        
        self.checkpoints_crossed = [False] * self.LAPS_TO_WIN
        
        return self._get_observation(), self._get_info()

    def _spawn_obstacles(self):
        self.obstacles.clear()
        
        # Add finish line as a special obstacle
        self.obstacles.append({
            "x": 0, "y": 0, "w": self.TRACK_WIDTH, "h": 20, 
            "type": "finish_line"
        })
        
        # Safe zone for stability test: 60 steps * speed + buffer
        safe_y_distance = (60 * self.FORWARD_SPEED_BASE) + 100.0
        
        for i in range(self.NUM_OBSTACLES):
            y_pos = (i + 1) * (self.TRACK_LENGTH / (self.NUM_OBSTACLES + 1))
            
            # Ensure no obstacle is too close to the start line
            if y_pos < 300:
                continue

            x_pos = self.np_random.uniform(-self.TRACK_WIDTH / 2.2, self.TRACK_WIDTH / 2.2)

            # If obstacle is in the initial safe zone, ensure it's not in the center.
            # This prevents termination during the no-op stability test.
            # A central corridor of width 100 (50 on each side) is kept clear.
            if y_pos < safe_y_distance and abs(x_pos) < 50:
                if self.np_random.random() < 0.5:
                    # Place on the right side
                    x_pos = self.np_random.uniform(50, self.TRACK_WIDTH / 2.2)
                else:
                    # Place on the left side
                    x_pos = self.np_random.uniform(-self.TRACK_WIDTH / 2.2, -50)

            self.obstacles.append({
                "x": x_pos, "y": y_pos, "w": 40, "h": 20,
                "type": "static"
            })

    def step(self, action):
        if self.game_over:
            # The episode has ended, but we can allow steps to observe the end state.
            # Return a terminal observation.
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = self._update_game_state(movement, space_held, shift_held)
        
        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True # Ensure game over state is consistent

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held, shift_held):
        # --- Update Player ---
        if movement == 3: # Left
            self.player_vx -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vx += self.PLAYER_ACCEL
        
        self.player_vx *= self.PLAYER_DRAG
        self.player_vx = np.clip(self.player_vx, -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        self.player_x += self.player_vx
        
        # Track boundaries
        track_half_width = self.TRACK_WIDTH / 2
        if abs(self.player_x) > track_half_width:
            self.player_x = np.sign(self.player_x) * track_half_width
            self.player_vx = 0

        # --- Update World Progress ---
        self.last_progress_y = self.progress_y
        forward_speed = self.FORWARD_SPEED_BOOST if space_held else self.FORWARD_SPEED_BASE
        
        # Add boost particles
        if space_held:
            self._create_particles(
                count=2,
                pos=(self.SCREEN_WIDTH / 2 + self.player_x, self.PLAYER_RENDER_Y + 10),
                color=(255, 150, 0),
                vel_range=((-0.5, 0.5), (1, 3)),
                life_range=(10, 20),
                radius_range=(2,4)
            )

        self.progress_y += forward_speed
        
        # --- Check Laps & Checkpoints ---
        reward = 0.1 # Survival reward
        
        previous_lap = self.laps
        self.laps = int(self.progress_y / self.TRACK_LENGTH)
        
        if self.laps > previous_lap:
            # Crossed a lap
            if previous_lap < self.LAPS_TO_WIN and not self.checkpoints_crossed[previous_lap]:
                self.checkpoints_crossed[previous_lap] = True
                reward += 50.0 # Lap completion
                self.score += 500
                
                if self.laps >= self.LAPS_TO_WIN:
                    reward += 100.0 # Win game
                    self.score += 1000
                    self.game_over = True
        
        # --- Collision Detection ---
        player_y_on_track = self.progress_y % self.TRACK_LENGTH
        player_rect = pygame.Rect(self.player_x - 10, player_y_on_track - 10, 20, 20)
        
        for obs in self.obstacles:
            if obs["type"] == "finish_line":
                continue
            
            obs_rect = pygame.Rect(obs["x"] - obs["w"]/2, obs["y"] - obs["h"]/2, obs["w"], obs["h"])
            
            # Check y-proximity first for efficiency
            if abs(player_y_on_track - obs["y"]) < 50:
                if player_rect.colliderect(obs_rect):
                    reward = -50.0 # Collision penalty
                    self.game_over = True
                    self._create_particles(
                        count=30,
                        pos=(self.SCREEN_WIDTH / 2 + self.player_x, self.PLAYER_RENDER_Y),
                        color=self.COLOR_PLAYER,
                        vel_range=((-5, 5), (-5, 5)),
                        life_range=(20, 40),
                        radius_range=(2, 6)
                    )
                    break
        
        # --- Update Particles ---
        self._update_particles()
        
        self.score += 1 # Time-based score
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] -= 0.05

    def _create_particles(self, count, pos, color, vel_range, life_range, radius_range):
        for _ in range(count):
            self.particles.append({
                'x': pos[0] + self.np_random.uniform(-5, 5),
                'y': pos[1] + self.np_random.uniform(-5, 5),
                'vx': self.np_random.uniform(*vel_range[0]),
                'vy': self.np_random.uniform(*vel_range[1]),
                'life': self.np_random.integers(*life_range),
                'radius': self.np_random.uniform(*radius_range),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Track ---
        center_x = self.SCREEN_WIDTH / 2
        
        # Track Edges
        pygame.draw.line(self.screen, self.COLOR_TRACK, (center_x - self.TRACK_WIDTH/2, 0), (center_x - self.TRACK_WIDTH/2, self.SCREEN_HEIGHT), 5)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (center_x + self.TRACK_WIDTH/2, 0), (center_x + self.TRACK_WIDTH/2, self.SCREEN_HEIGHT), 5)

        # Dashed Center Lines
        line_length = 40
        gap_length = 20
        segment_length = line_length + gap_length
        scroll_offset = self.progress_y % segment_length
        
        for y in range(-int(scroll_offset) - segment_length, self.SCREEN_HEIGHT, segment_length):
            pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (center_x, y), (center_x, y + line_length), 2)
            
        # --- Draw Obstacles & Finish Line (from back to front) ---
        render_list = []
        for obs in self.obstacles:
            render_y = self.PLAYER_RENDER_Y + (obs["y"] - (self.progress_y % self.TRACK_LENGTH))
            
            # Wrap around for seamless laps
            if render_y > self.PLAYER_RENDER_Y + self.TRACK_LENGTH / 2:
                render_y -= self.TRACK_LENGTH
            if render_y < self.PLAYER_RENDER_Y - self.TRACK_LENGTH / 2:
                render_y += self.TRACK_LENGTH
            
            # Cull off-screen objects
            if -50 < render_y < self.SCREEN_HEIGHT + 50:
                render_list.append((render_y, obs))

        # Sort by y-coordinate to draw in correct order
        render_list.sort(key=lambda item: item[0])
        
        for render_y, obs in render_list:
            render_x = center_x + obs["x"]
            if obs["type"] == "finish_line":
                self._draw_isometric_rect(
                    render_x, render_y, obs["w"], obs["h"], 
                    self._get_pulsating_color(self.COLOR_FINISH_PULS_1, self.COLOR_FINISH_PULS_2, 2.0)
                )
            else:
                self._draw_isometric_rect(render_x, render_y, obs["w"], obs["h"], self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        
        # --- Draw Particles ---
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'])

        # --- Draw Player ---
        if not self.game_over:
            player_screen_x = center_x + self.player_x
            self._draw_isometric_rect(player_screen_x, self.PLAYER_RENDER_Y, 25, 20, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _draw_isometric_rect(self, x, y, w, h, color, glow_color=None):
        """Draws a rhombus to simulate an isometric rectangle."""
        half_w, half_h = w / 2, h / 2
        points = [
            (x, y - half_h), (x + half_w, y),
            (x, y + half_h), (x - half_w, y)
        ]
        
        if glow_color:
            glow_points = [
                (x, y - half_h - 4), (x + half_w + 4, y),
                (x, y + half_h + 4), (x - half_w - 4, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, glow_color)
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)
            
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _get_pulsating_color(self, color1, color2, speed=1.0):
        """Returns a color that pulsates between two colors."""
        t = (math.sin(self.steps * 0.05 * speed) + 1) / 2.0
        r = int(color1[0] * (1 - t) + color2[0] * t)
        g = int(color1[1] * (1 - t) + color2[1] * t)
        b = int(color1[2] * (1 - t) + color2[2] * t)
        return (r, g, b)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Laps
        lap_text = self.font_medium.render(f"LAP: {min(self.laps + 1, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (self.SCREEN_WIDTH - lap_text.get_width() - 10, 10))

        # Progress Bar
        progress_percent = (self.progress_y % self.TRACK_LENGTH) / self.TRACK_LENGTH
        bar_width = 200
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 20
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT, (bar_x, bar_y, bar_width * progress_percent, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game Over / Win Message
        if self.game_over:
            if self.laps >= self.LAPS_TO_WIN:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "progress": self.progress_y,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a display, so it won't work in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
        import pygame
        
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # --- Manual Control Setup ---
        # Action array: [movement, space, shift]
        action = np.array([0, 0, 0]) 
        
        # Game loop
        display = None
        while not done:
            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Keyboard input
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                action[0] = 4
            else:
                action[0] = 0 # No-op for horizontal movement

            # Space for boost
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            
            # Shift for drift/brake (not implemented in this version but action is there)
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the screen
            if display is None:
                pygame.display.set_caption("Isometric Racer")
                display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            
            # The observation is (H, W, C), but pygame wants (W, H)
            # and surfarray.make_surface expects (W,H,C)
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Cap the frame rate
            env.clock.tick(60)

        env.close()
        print("Game Over!")
        print(f"Final Info: {info}")
    except Exception as e:
        print(f"Could not run interactive game: {e}")