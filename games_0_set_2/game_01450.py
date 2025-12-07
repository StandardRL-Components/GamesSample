
# Generated: 2025-08-27T17:10:57.050406
# Source Brief: brief_01450.md
# Brief Index: 1450

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the drawing cursor. "
        "Press SPACE to set the start of a line, then move the cursor and "
        "press SHIFT to draw the line. Your rider will slide on these lines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based sledding game. Draw lines on the screen to create slopes "
        "and ramps, guiding your rider from the start to the finish line. "
        "Avoid crashing out of bounds!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # Class attributes for persistent difficulty scaling
    _successful_runs = 0
    _difficulty_level = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1200
        self.MAX_DRAWN_LINES = 8
        self.RIDER_RADIUS = 8
        self.CURSOR_SPEED = 8
        self.GRAVITY = 0.25
        self.FRICTION = 0.995

        # --- Colors ---
        self.COLOR_BG_SKY = (135, 206, 235)
        self.COLOR_BG_GROUND = (211, 211, 211)
        self.COLOR_RIDER = (255, 69, 0)
        self.COLOR_RIDER_GLOW = (255, 140, 0, 100)
        self.COLOR_TERRAIN = (50, 50, 50)
        self.COLOR_DRAWN_LINE = (10, 10, 10)
        self.COLOR_START = (0, 200, 0)
        self.COLOR_FINISH = (200, 0, 0)
        self.COLOR_CHECKPOINT = (255, 215, 0)
        self.COLOR_CURSOR = (0, 0, 255, 150)
        self.COLOR_DRAFT_LINE = (0, 0, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.drawing_start_pos = None
        self.drawn_lines = []
        self.terrain_points = []
        self.all_lines = []
        self.particles = []
        self.trail = deque(maxlen=25)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.checkpoint_reached = False
        self.finish_line_x = 0
        self.checkpoint_x = 0
        self.start_pos_y = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_terrain()
        
        self.start_pos = pygame.math.Vector2(50, self.start_pos_y - self.RIDER_RADIUS * 2)
        self.rider_pos = self.start_pos.copy()
        self.rider_vel = pygame.math.Vector2(2, 0) # Initial push

        self.cursor_pos = pygame.math.Vector2(self.W / 2, self.H / 2)
        self.drawing_start_pos = None
        self.drawn_lines = []
        self._update_all_lines()

        self.particles = []
        self.trail.clear()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.checkpoint_reached = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        prev_rider_pos = self.rider_pos.copy()
        if not self.game_over:
            self._update_physics()
            self.trail.append(self.rider_pos.copy())

        self._update_particles()
        self.steps += 1
        
        reward, terminated = self._calculate_reward_and_termination(prev_rider_pos)
        self.score += reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_terrain(self):
        self.terrain_points = []
        y = self.H * 0.5
        self.start_pos_y = y
        
        # Difficulty scales amplitude and frequency of hills
        base_amp = 40
        base_freq = 0.01
        difficulty_mod = 1.0 + (GameEnv._difficulty_level * 0.2)
        
        amplitude = base_amp * difficulty_mod
        frequency = base_freq * difficulty_mod
        num_hills = 3 + GameEnv._difficulty_level

        x = 0
        while x < self.W + 50:
            y_offset = 0
            for i in range(1, num_hills + 1):
                y_offset += (amplitude / i) * math.sin(x * frequency * i * 0.5 + i)
            
            self.terrain_points.append(pygame.math.Vector2(x, self.H * 0.6 + y_offset))
            x += 20
        
        self.finish_line_x = self.terrain_points[-5].x
        self.checkpoint_x = self.W * 0.55

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.W)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.H)

        # Start drawing line on space press
        if space_held and not self.prev_space_held:
            self.drawing_start_pos = self.cursor_pos.copy()
            # Sound: UI_Select

        # Finish drawing line on shift press
        if shift_held and not self.prev_shift_held and self.drawing_start_pos:
            new_line = (self.drawing_start_pos, self.cursor_pos.copy())
            self.drawn_lines.append(new_line)
            if len(self.drawn_lines) > self.MAX_DRAWN_LINES:
                self.drawn_lines.pop(0)
            self.drawing_start_pos = None
            self._update_all_lines()
            # Sound: Line_Draw

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

    def _update_physics(self):
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        
        # Move
        self.rider_pos += self.rider_vel

        # Collision detection and response
        highest_ground_y = float('inf')
        ground_line = None
        
        for p1, p2 in self.all_lines:
            if p1.x == p2.x: continue # Skip vertical lines
            
            # Check if rider is horizontally within the line segment
            if min(p1.x, p2.x) - self.RIDER_RADIUS < self.rider_pos.x < max(p1.x, p2.x) + self.RIDER_RADIUS:
                # Interpolate line's y at rider's x
                t = (self.rider_pos.x - p1.x) / (p2.x - p1.x)
                line_y = p1.y + t * (p2.y - p1.y)

                # If rider is at or below the line and this is the highest line found so far
                if self.rider_pos.y + self.RIDER_RADIUS >= line_y and line_y < highest_ground_y:
                    highest_ground_y = line_y
                    ground_line = (p1, p2)
        
        if ground_line:
            p1, p2 = ground_line
            self.rider_pos.y = highest_ground_y - self.RIDER_RADIUS

            # Slide physics
            line_vec = (p2 - p1).normalize()
            dot_product = self.rider_vel.dot(line_vec)
            self.rider_vel = line_vec * dot_product
            self.rider_vel *= self.FRICTION
            
            # Create sparks on contact
            if self.rider_vel.length() > 2:
                self._create_sparks(self.rider_pos + pygame.math.Vector2(0, self.RIDER_RADIUS), 2)
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1 # lifetime--

    def _create_sparks(self, pos, count):
        for _ in range(count):
            vel = pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 0))
            lifetime = random.randint(10, 20)
            self.particles.append([pos.copy(), vel, lifetime])
            # Sound: Spark

    def _update_all_lines(self):
        self.all_lines = []
        for i in range(len(self.terrain_points) - 1):
            self.all_lines.append((self.terrain_points[i], self.terrain_points[i+1]))
        self.all_lines.extend(self.drawn_lines)

    def _calculate_reward_and_termination(self, prev_rider_pos):
        reward = 0.0
        terminated = False

        # Reward for forward progress
        dx = self.rider_pos.x - prev_rider_pos.x
        if dx > 0:
            reward += dx * 0.1
        else:
            reward -= 0.01

        # Checkpoint reward
        if not self.checkpoint_reached and self.rider_pos.x > self.checkpoint_x:
            self.checkpoint_reached = True
            reward += 5.0
            # Sound: Checkpoint_Get

        # Termination conditions
        if self.rider_pos.x >= self.finish_line_x:
            reward += 100.0
            terminated = True
            GameEnv._successful_runs += 1
            if GameEnv._successful_runs > 0 and GameEnv._successful_runs % 5 == 0:
                GameEnv._difficulty_level = min(10, GameEnv._difficulty_level + 1)
            # Sound: Win_Jingle

        if not (0 < self.rider_pos.x < self.W and -self.RIDER_RADIUS < self.rider_pos.y < self.H):
            reward -= 50.0
            terminated = True
            # Sound: Crash_Sound

        if self.steps >= self.MAX_STEPS:
            terminated = True
            # No extra penalty, just end of episode

        return reward, terminated

    def _get_observation(self):
        # --- Background Gradient ---
        for y in range(self.H):
            interp = y / self.H
            color = (
                self.COLOR_BG_SKY[0] * (1 - interp) + self.COLOR_BG_GROUND[0] * interp,
                self.COLOR_BG_SKY[1] * (1 - interp) + self.COLOR_BG_GROUND[1] * interp,
                self.COLOR_BG_SKY[2] * (1 - interp) + self.COLOR_BG_GROUND[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))

        # --- Game Elements ---
        self._render_game()
        self._render_ui()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Start/Finish/Checkpoint Lines ---
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos.x, 0), (self.start_pos.x, self.H), 3)
        pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (self.checkpoint_x, 0), (self.checkpoint_x, self.H), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.H), 3)

        # --- Terrain ---
        if len(self.terrain_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, self.terrain_points, 3)

        # --- Drawn Lines ---
        for p1, p2 in self.drawn_lines:
            pygame.draw.aaline(self.screen, self.COLOR_DRAWN_LINE, p1, p2, 4)

        # --- Rider Trail ---
        if len(self.trail) > 2:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / len(self.trail)))
                color = (*self.COLOR_RIDER_GLOW[:3], alpha)
                pygame.draw.line(self.screen, color, self.trail[i], self.trail[i+1], max(1, int(self.RIDER_RADIUS * (i / len(self.trail)))))

        # --- Particles ---
        for p in self.particles:
            pos, _, lifetime = p
            alpha = max(0, min(255, lifetime * 15))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 2, (255, 255, 255, alpha))

        # --- Rider ---
        rider_center = (int(self.rider_pos.x), int(self.rider_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, *rider_center, self.RIDER_RADIUS + 3, self.COLOR_RIDER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, *rider_center, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, *rider_center, self.RIDER_RADIUS, self.COLOR_RIDER)

        # --- Drawing UI ---
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_CURSOR)
        if self.drawing_start_pos:
            pygame.draw.aaline(self.screen, self.COLOR_DRAFT_LINE, self.drawing_start_pos, self.cursor_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(self.drawing_start_pos.x), int(self.drawing_start_pos.y), 5, self.COLOR_CURSOR)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, pos, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Time and Distance
        time_text = f"Time: {self.steps}"
        dist_text = f"Distance: {int(self.rider_pos.x)} / {int(self.finish_line_x)}"
        draw_text(time_text, self.font_small, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        draw_text(dist_text, self.font_small, (self.W - 200, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over message
        if self.game_over:
            if self.rider_pos.x >= self.finish_line_x:
                msg = "FINISH!"
            else:
                msg = "CRASHED!"
            draw_text(msg, self.font_large, (self.W/2 - 100, self.H/2 - 30), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
            "lines_drawn": len(self.drawn_lines)
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
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Human Playable Demo ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for display
    screen_display = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Sled Rider")
    clock = pygame.time.Clock()

    terminated = False
    running = True
    while running:
        if terminated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            terminated = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        clock.tick(env.FPS)

    env.close()