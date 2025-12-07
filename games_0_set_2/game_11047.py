import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:31:44.027690
# Source Brief: brief_01047.md
# Brief Index: 1047
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Ant Squad'.

    The player controls a squad of ants to gather 120 crumbs within 90 seconds.
    The action space is MultiDiscrete, allowing simultaneous movement and
    cycling through the controllable ants.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a squad of ants to gather food crumbs as quickly as possible. "
        "Switch between ants and use their collective speed to meet the quota before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected ant. "
        "Press space to cycle to the next ant, and shift to cycle to the previous ant."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 90
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    WIN_SCORE = 120
    INITIAL_CRUMBS = 20
    NUM_ANTS = 5

    # Colors
    COLOR_BG = (50, 40, 30)
    COLOR_ANT_BASE = (150, 100, 50)
    COLOR_ANT_LIGHTEN = (25, 25, 20) # Amount to lighten per crumb
    COLOR_CRUMB = (255, 255, 240)
    COLOR_UI_TEXT = (240, 240, 220)
    COLOR_SELECTED_AURA = (255, 255, 0)

    # Game entity properties
    ANT_RADIUS = 8
    ANT_BASE_SPEED = 3.5
    ANT_SPEED_BONUS_PER_CRUMB = 0.05
    ANT_CAPACITY = 3
    CRUMB_RADIUS = 3
    MIN_SPAWN_DISTANCE = 50 # Min distance from other entities

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- Game State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ants = []
        self.crumbs = []
        self.selected_ant_index = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_ant_index = 0
        self.previous_space_held = False
        self.previous_shift_held = False

        # --- Initialize Ants ---
        self.ants = []
        for i in range(self.NUM_ANTS):
            # Spawn ants in a rough circle near the center
            angle = (2 * math.pi / self.NUM_ANTS) * i
            spawn_x = self.SCREEN_WIDTH / 2 + math.cos(angle) * 100
            spawn_y = self.SCREEN_HEIGHT / 2 + math.sin(angle) * 100
            self.ants.append(self._Ant(pos=pygame.math.Vector2(spawn_x, spawn_y),
                                       radius=self.ANT_RADIUS,
                                       base_speed=self.ANT_BASE_SPEED,
                                       capacity=self.ANT_CAPACITY,
                                       color_base=self.COLOR_ANT_BASE,
                                       color_lighten=self.COLOR_ANT_LIGHTEN))

        # --- Initialize Crumbs ---
        self.crumbs = []
        for _ in range(self.INITIAL_CRUMBS):
            self._spawn_crumb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack and Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle ant selection (on key press, not hold)
        if space_held and not self.previous_space_held:
            self.selected_ant_index = (self.selected_ant_index + 1) % self.NUM_ANTS
            # sfx: UI_Cycle_Forward.wav
        if shift_held and not self.previous_shift_held:
            self.selected_ant_index = (self.selected_ant_index - 1 + self.NUM_ANTS) % self.NUM_ANTS
            # sfx: UI_Cycle_Backward.wav

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

        # Handle movement for the selected ant
        selected_ant = self.ants[self.selected_ant_index]
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1  # Up
        elif movement == 2: move_vector.y = 1   # Down
        elif movement == 3: move_vector.x = -1  # Left
        elif movement == 4: move_vector.x = 1   # Right
        
        if move_vector.length_squared() > 0:
            move_vector.normalize_ip()
            selected_ant.move(move_vector, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # --- Game Logic Update ---
        reward = 0
        
        # Check for crumb collection for all ants
        for ant in self.ants:
            for i in range(len(self.crumbs) - 1, -1, -1):
                crumb_pos = self.crumbs[i]
                distance = ant.pos.distance_to(crumb_pos)
                if distance < ant.radius + self.CRUMB_RADIUS:
                    if ant.collect_crumb():
                        self.score += 1
                        reward += 0.1
                        del self.crumbs[i]
                        self._spawn_crumb()
                        # sfx: Collect_Crumb.wav
                        break # Ant can only collect one crumb per step

        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: Victory.wav
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: Failure.wav

        # Gymnasium API requires a boolean for truncated, not related to termination
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "selected_ant": self.selected_ant_index
        }
        
    def _render_game(self):
        # Render crumbs
        for crumb_pos in self.crumbs:
            pygame.gfxdraw.filled_circle(self.screen, int(crumb_pos.x), int(crumb_pos.y), self.CRUMB_RADIUS, self.COLOR_CRUMB)

        # Render ants
        for i, ant in enumerate(self.ants):
            # Highlight selected ant with a glowing aura
            if i == self.selected_ant_index and not self.game_over:
                aura_radius = int(ant.radius * 1.8)
                aura_color = (*self.COLOR_SELECTED_AURA, 80) # RGBA with alpha
                # Create a temporary surface for the aura for transparency
                aura_surface = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(aura_surface, aura_radius, aura_radius, aura_radius, aura_color)
                self.screen.blit(aura_surface, (int(ant.pos.x - aura_radius), int(ant.pos.y - aura_radius)))

            # Draw ant body
            pygame.gfxdraw.aacircle(self.screen, int(ant.pos.x), int(ant.pos.y), ant.radius, ant.color)
            pygame.gfxdraw.filled_circle(self.screen, int(ant.pos.x), int(ant.pos.y), ant.radius, ant.color)
            
            # Draw tiny ant legs for visual flair
            angle_offset = (self.steps % 30) / 30.0 * math.pi * 2
            for j in range(6):
                angle = j * (math.pi / 3) + angle_offset
                start_pos = ant.pos
                end_pos = ant.pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * (ant.radius + 2)
                pygame.draw.aaline(self.screen, ant.color, start_pos, end_pos)


    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"Crumbs: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Render game over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "VICTORY!"
                color = (150, 255, 150)
            else:
                msg = "TIME UP!"
                color = (255, 150, 150)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _spawn_crumb(self):
        # Find a valid spawn location for a new crumb
        padding = 10
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
            )
            
            # Check distance to other crumbs
            too_close = False
            for crumb_pos in self.crumbs:
                if pos.distance_to(crumb_pos) < self.MIN_SPAWN_DISTANCE:
                    too_close = True
                    break
            if too_close: continue

            # Check distance to ants
            for ant in self.ants:
                if pos.distance_to(ant.pos) < self.MIN_SPAWN_DISTANCE:
                    too_close = True
                    break
            if too_close: continue
            
            # If we reach here, the position is valid
            self.crumbs.append(pos)
            break

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    class _Ant:
        """Helper class to manage individual ant state."""
        def __init__(self, pos, radius, base_speed, capacity, color_base, color_lighten):
            self.pos = pos
            self.radius = radius
            self.base_speed = base_speed
            self.capacity = capacity
            self.color_base = color_base
            self.color_lighten = color_lighten
            
            self.carried_crumbs = 0
            self.speed = base_speed
            self.color = color_base

        def move(self, direction_vector, bounds):
            self.pos += direction_vector * self.speed
            # Clamp position to screen bounds
            self.pos.x = max(self.radius, min(self.pos.x, bounds[0] - self.radius))
            self.pos.y = max(self.radius, min(self.pos.y, bounds[1] - self.radius))
        
        def collect_crumb(self):
            if self.carried_crumbs < self.capacity:
                self.carried_crumbs += 1
                self._update_properties()
                return True
            return False

        def _update_properties(self):
            # Update speed
            self.speed = self.base_speed * (1 + GameEnv.ANT_SPEED_BONUS_PER_CRUMB * self.carried_crumbs)
            
            # Update color
            new_r = min(255, self.color_base[0] + self.color_lighten[0] * self.carried_crumbs)
            new_g = min(255, self.color_base[1] + self.color_lighten[1] * self.carried_crumbs)
            new_b = min(255, self.color_base[2] + self.color_lighten[2] * self.carried_crumbs)
            self.color = (new_r, new_g, new_b)

if __name__ == '__main__':
    # --- Manual Play Example ---
    # To run this, you need to unset the dummy video driver
    # Comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # at the top of the file.
    
    # Check if we can run with a display
    try:
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Setup for manual play window
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Ant Squad - Manual Control")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Arrows: Move | Space: Next Ant | Shift: Prev Ant | Q: Quit")

        while running:
            # --- Action Mapping for Human ---
            movement = 0 # None
            space_held = 0
            shift_held = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False

            action = [movement, space_held, shift_held]
            
            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Render to Screen ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()
                total_reward = 0

            clock.tick(GameEnv.FPS)

        env.close()
    except pygame.error as e:
        print(f"\nCould not run manual play example: {e}")
        print("This is expected if you are in a headless environment.")
        print("The environment code is still valid for training.")
        # Re-set the dummy driver for headless environments
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        # Create a dummy env to ensure it initializes correctly headless
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("Headless environment initialization successful.")