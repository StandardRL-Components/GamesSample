import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:09:01.947088
# Source Brief: brief_02611.md
# Brief Index: 2611
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a frog on a course of lily pads.
    The goal is to reach the end of the course while avoiding predators (birds) and
    falling into the water. The frog can change size to dodge or make bigger leaps.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0=none, 1=up(unused), 2=down(unused), 3=left, 4=right)
    - `actions[1]`: Space button (0=released, 1=held) -> Jump
    - `actions[2]`: Shift button (0=released, 1=held) -> Shrink / Combine with Space to Grow

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Reward Structure:**
    - +0.01 for each pixel moved forward.
    - -0.01 for each pixel moved backward.
    - +10 for successfully landing a jump on a new lily pad.
    - -100 for being caught by a bird (terminal).
    - -100 for falling in the water (terminal).
    - +200 for reaching the finish line (terminal).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a frog on a course of lily pads. Jump to progress, dodge birds, and manage your energy to reach the finish line."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move. Press space to jump. Hold shift to shrink, or hold shift and space to grow for a bigger jump."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    COURSE_LENGTH = 5000
    MAX_EPISODE_STEPS = 2500

    # Colors
    COLOR_BG = (52, 152, 219)        # Belizae Hole Blue
    COLOR_WATER_WAVE = (41, 128, 185) # Belize Hole Darker
    COLOR_PAD = (39, 174, 96)         # Nephritis Green
    COLOR_PAD_DARK = (22, 160, 133)   # Green Sea
    COLOR_FROG = (211, 84, 0)         # Pumpkin Orange
    COLOR_FROG_GLOW = (230, 126, 34, 100) # Carrot Orange (with alpha)
    COLOR_BIRD = (192, 57, 43)        # Pomegranate Red
    COLOR_BIRD_SHADOW = (44, 62, 80, 150) # Wet Asphalt (with alpha)
    COLOR_UI_TEXT = (236, 240, 241)   # Clouds White
    COLOR_UI_BAR = (241, 196, 15)     # Sun Flower Yellow
    COLOR_UI_BAR_BG = (127, 140, 141) # Asbestos Gray

    # Physics & Gameplay
    GRAVITY = 0.4
    FROG_MOVE_SPEED = 3.0
    FROG_JUMP_FORCE = -10.0
    FROG_MIN_SIZE = 8
    FROG_NORMAL_SIZE = 15
    FROG_MAX_SIZE = 22
    SIZE_CHANGE_RATE = 0.5
    GROW_COST = 1.0

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
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 50, bold=True)
        
        # Persistent state (survives reset)
        self.jump_power_enhancement = 1.0

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.camera_x = 0.0
        self.max_progress = 0.0

        self.frog = {}
        self.lily_pads = []
        self.birds = []
        self.particles = []
        
        # self.reset() # reset() is called by the wrapper/runner
        # self.validate_implementation() # Validation is done by tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0.0
        self.max_progress = 0.0
        
        self.frog = {
            "x": 100.0, "y": self.SCREEN_HEIGHT - 50,
            "vx": 0.0, "vy": 0.0,
            "size": self.FROG_NORMAL_SIZE, "target_size": self.FROG_NORMAL_SIZE,
            "on_pad": True, "last_pad_idx": 0,
            "leg_muscle": 100.0,
        }

        self._generate_course()
        self._generate_birds()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- 1. Handle Actions & Update Frog State ---
        self._handle_input(movement, space_held, shift_held)
        self._update_frog_physics()
        old_x = self.frog['x']

        # --- 2. Update Environment ---
        self._update_entities()
        self._update_camera()

        # --- 3. Collision Detection & State Checks ---
        landed_on_new_pad = self._check_collisions()
        self._check_boundaries()

        # --- 4. Calculate Reward ---
        progress_diff = self.frog['x'] - old_x
        if self.frog['x'] > self.max_progress:
            reward += (self.frog['x'] - self.max_progress) * 0.01
            self.max_progress = self.frog['x']
        elif progress_diff < 0:
            reward += progress_diff * 0.01

        if landed_on_new_pad:
            reward += 10.0
            # Sfx: land_splash.wav
        
        self.score += reward

        # --- 5. Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = False
        if terminated:
            if self.frog['x'] >= self.COURSE_LENGTH:
                reward += 200.0 # Reached finish line
                self.game_over_message = "COURSE COMPLETE!"
                self.jump_power_enhancement += 0.05 # Persistent upgrade
            elif self.frog['y'] > self.SCREEN_HEIGHT:
                reward -= 100.0 # Fell in water
                self.game_over_message = "GAME OVER"
                # Sfx: splash_death.wav
            else: # Bird collision
                reward -= 100.0
                self.game_over_message = "GAME OVER"
                # Sfx: bird_squawk.wav
            self.score += reward

        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS and not terminated:
            truncated = True
            terminated = True # Gymnasium standard: truncated episodes are also terminated
            self.game_over = True
            self.game_over_message = "TIME UP"
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Size change
        if shift_held:
            if space_held and self.frog['leg_muscle'] > 0:
                self.frog['target_size'] = self.FROG_MAX_SIZE
                self.frog['leg_muscle'] = max(0, self.frog['leg_muscle'] - self.GROW_COST)
            else:
                self.frog['target_size'] = self.FROG_MIN_SIZE
        else:
            self.frog['target_size'] = self.FROG_NORMAL_SIZE

        # Horizontal Movement
        if movement == 3: # Left
            self.frog['vx'] = -self.FROG_MOVE_SPEED
        elif movement == 4: # Right
            self.frog['vx'] = self.FROG_MOVE_SPEED
        else:
            self.frog['vx'] = 0

        # Jumping
        if space_held and not shift_held and self.frog['on_pad']:
            self.frog['vy'] = self.FROG_JUMP_FORCE * self.jump_power_enhancement
            self.frog['on_pad'] = False
            # Sfx: jump.wav
            for _ in range(10):
                self.particles.append(self._create_particle(self.frog['x'], self.frog['y'] + self.frog['size'], (255,255,255), 2, 5))

    def _update_frog_physics(self):
        # Apply velocity
        self.frog['x'] += self.frog['vx']
        
        # Apply gravity if not on a pad
        if not self.frog['on_pad']:
            self.frog['y'] += self.frog['vy']
            self.frog['vy'] += self.GRAVITY
        else:
            self.frog['vy'] = 0
            # Regenerate leg muscle on pad
            self.frog['leg_muscle'] = min(100.0, self.frog['leg_muscle'] + 0.2)

        # Smooth size change
        if self.frog['size'] != self.frog['target_size']:
            diff = self.frog['target_size'] - self.frog['size']
            self.frog['size'] += np.sign(diff) * min(abs(diff), self.SIZE_CHANGE_RATE)

    def _update_entities(self):
        # Update birds
        bird_speed_multiplier = 1.0 + (self.steps / 500) * 0.05
        for bird in self.birds:
            bird['x'] += bird['vx'] * bird_speed_multiplier
            if bird['x'] < -50 or bird['x'] > self.COURSE_LENGTH + 50:
                bird['x'] = self.COURSE_LENGTH + 50 if bird['vx'] < 0 else -50

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _check_collisions(self):
        # Frog vs Lily Pads
        self.frog['on_pad'] = False
        landed_on_new_pad = False
        for i, pad in enumerate(self.lily_pads):
            is_above = self.frog['y'] + self.frog['size'] > pad['y'] - 5 and self.frog['y'] + self.frog['size'] < pad['y'] + 10
            is_horizontally_aligned = self.frog['x'] > pad['x'] - pad['width'] / 2 and self.frog['x'] < pad['x'] + pad['width'] / 2
            
            if self.frog['vy'] > 0 and is_above and is_horizontally_aligned:
                self.frog['on_pad'] = True
                self.frog['y'] = pad['y'] - self.frog['size']
                self.frog['vy'] = 0
                if self.frog['last_pad_idx'] != i:
                    landed_on_new_pad = True
                    self.frog['last_pad_idx'] = i
                    for _ in range(20):
                        self.particles.append(self._create_particle(self.frog['x'], pad['y'], self.COLOR_WATER_WAVE, 1, 4, life=40))
                break
        return landed_on_new_pad

    def _check_boundaries(self):
        # Prevent moving backwards off screen
        self.frog['x'] = max(self.frog['size'], self.frog['x'])
        # Water and bird collisions handled in termination check

    def _check_termination(self):
        # Fall in water
        if self.frog['y'] > self.SCREEN_HEIGHT:
            self.game_over = True
            return True

        # Caught by bird
        frog_rect = pygame.Rect(self.frog['x'] - self.frog['size'], self.frog['y'] - self.frog['size'], self.frog['size']*2, self.frog['size']*2)
        for bird in self.birds:
            bird_rect = pygame.Rect(bird['x'] - 20, bird['y'] - 10, 40, 20)
            if frog_rect.colliderect(bird_rect):
                self.game_over = True
                return True

        # Reached finish line
        if self.frog['x'] >= self.COURSE_LENGTH:
            self.game_over = True
            return True
            
        return False

    def _update_camera(self):
        target_camera_x = self.frog['x'] - self.SCREEN_WIDTH / 3
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, self.camera_x)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw water waves
        for i in range(0, self.SCREEN_HEIGHT, 20):
            amplitude = 3
            frequency = 0.01
            offset = self.steps * 0.5
            points = []
            for x in range(0, self.SCREEN_WIDTH + 1, 10):
                y = i + math.sin(x * frequency + offset) * amplitude
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_WATER_WAVE, False, points, 1)

        # Draw finish line
        finish_x = self.COURSE_LENGTH - self.camera_x
        if finish_x < self.SCREEN_WIDTH + 50:
            pygame.draw.line(self.screen, (255,255,255), (finish_x, 0), (finish_x, self.SCREEN_HEIGHT), 5)
            pygame.draw.line(self.screen, (0,0,0), (finish_x-5, 0), (finish_x-5, self.SCREEN_HEIGHT), 2)
            pygame.draw.line(self.screen, (0,0,0), (finish_x+5, 0), (finish_x+5, self.SCREEN_HEIGHT), 2)

        # Draw lily pads
        for pad in self.lily_pads:
            draw_x = pad['x'] - self.camera_x
            if -pad['width'] < draw_x < self.SCREEN_WIDTH + pad['width']:
                pygame.gfxdraw.filled_ellipse(self.screen, int(draw_x), int(pad['y']), int(pad['width'] / 2), int(pad['width'] / 4), self.COLOR_PAD)
                pygame.gfxdraw.ellipse(self.screen, int(draw_x), int(pad['y']), int(pad['width'] / 2), int(pad['width'] / 4), self.COLOR_PAD_DARK)

        # Draw particles
        for p in self.particles:
            draw_x = p['x'] - self.camera_x
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(draw_x), int(p['y']), int(p['size']), color)

        # Draw bird shadows
        for bird in self.birds:
            draw_x = bird['x'] - self.camera_x
            shadow_y = self.SCREEN_HEIGHT - 30
            pygame.gfxdraw.filled_ellipse(self.screen, int(draw_x), shadow_y, 25, 8, self.COLOR_BIRD_SHADOW)

        # Draw frog
        frog_draw_x = int(self.frog['x'] - self.camera_x)
        frog_draw_y = int(self.frog['y'])
        frog_size = int(self.frog['size'])
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, frog_draw_x, frog_draw_y, int(frog_size * 1.4), self.COLOR_FROG_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, frog_draw_x, frog_draw_y, frog_size, self.COLOR_FROG)
        pygame.gfxdraw.aacircle(self.screen, frog_draw_x, frog_draw_y, frog_size, (0,0,0))
        # Eyes
        eye_offset_x = int(frog_size * 0.4)
        eye_offset_y = int(frog_size * 0.3)
        eye_size = int(frog_size * 0.2)
        pygame.draw.circle(self.screen, (255,255,255), (frog_draw_x - eye_offset_x, frog_draw_y - eye_offset_y), eye_size)
        pygame.draw.circle(self.screen, (255,255,255), (frog_draw_x + eye_offset_x, frog_draw_y - eye_offset_y), eye_size)
        pygame.draw.circle(self.screen, (0,0,0), (frog_draw_x - eye_offset_x, frog_draw_y - eye_offset_y), int(eye_size * 0.5))
        pygame.draw.circle(self.screen, (0,0,0), (frog_draw_x + eye_offset_x, frog_draw_y - eye_offset_y), int(eye_size * 0.5))

        # Draw birds
        for bird in self.birds:
            draw_x = int(bird['x'] - self.camera_x)
            draw_y = int(bird['y'])
            # Simple bird shape
            wing_y = draw_y + int(math.sin(self.steps * 0.2 + bird['x'] * 0.1) * 8)
            pygame.draw.polygon(self.screen, self.COLOR_BIRD, [(draw_x, draw_y), (draw_x - 20, wing_y), (draw_x + 20, wing_y)])


    def _render_ui(self):
        # Leg Muscle Bar
        frog_draw_x = int(self.frog['x'] - self.camera_x)
        bar_y = int(self.frog['y'] - self.frog['size'] - 20)
        bar_width = 50
        bar_height = 8
        fill_width = int((self.frog['leg_muscle'] / 100.0) * bar_width)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (frog_draw_x - bar_width//2, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (frog_draw_x - bar_width//2, bar_y, fill_width, bar_height))

        # Score and Distance Text
        dist_text = f"Distance: {int(self.frog['x'])} / {self.COURSE_LENGTH}"
        score_text = f"Score: {int(self.score)}"
        
        dist_surface = self.font_ui.render(dist_text, True, self.COLOR_UI_TEXT)
        score_surface = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(dist_surface, (10, 10))
        self.screen.blit(score_surface, (self.SCREEN_WIDTH - score_surface.get_width() - 10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0, 180))
        self.screen.blit(s, (0,0))
        
        text_surface = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "frog_x": self.frog['x'],
            "frog_y": self.frog['y'],
            "leg_muscle": self.frog['leg_muscle'],
            "jump_power": self.jump_power_enhancement,
        }

    def _generate_course(self):
        self.lily_pads = []
        x = 100
        y = self.SCREEN_HEIGHT - 40
        gap_increase_factor = 1.0 + (self.steps / 1000)
        
        while x < self.COURSE_LENGTH:
            width = self.np_random.integers(80, 150)
            self.lily_pads.append({'x': x, 'y': y, 'width': width})
            
            gap = self.np_random.integers(50, 150) * gap_increase_factor
            x += width / 2 + gap
            y = self.np_random.integers(self.SCREEN_HEIGHT - 150, self.SCREEN_HEIGHT - 30)

    def _generate_birds(self):
        self.birds = []
        for _ in range(5):
            self.birds.append({
                'x': self.np_random.uniform(0, self.COURSE_LENGTH),
                'y': self.np_random.uniform(50, self.SCREEN_HEIGHT / 2),
                'vx': self.np_random.choice([-2.5, 2.5])
            })

    def _create_particle(self, x, y, color, min_speed, max_speed, life=20):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(min_speed, max_speed)
        return {
            "x": x, "y": y,
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "size": self.np_random.integers(2, 5),
            "life": life, "max_life": life,
            "color": color
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    # To run, unset SDL_VIDEODRIVER or set it to a valid driver.
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
        print("Cannot run main in headless mode. Unset SDL_VIDEODRIVER to play manually.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        
        running = True
        terminated = False
        truncated = False
        
        # Pygame setup for manual play
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Frog Leap")
        clock = pygame.time.Clock()

        while running:
            # --- Action mapping for human player ---
            keys = pygame.key.get_pressed()
            
            movement = 0 # None
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    terminated = False
                    truncated = False

            if not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering ---
            # The observation is already a rendered frame, we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Run at 30 FPS

        env.close()