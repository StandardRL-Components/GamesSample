import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Navigate a shrinking planetary system by manipulating a prism's size
    to match planet-specific requirements, revealing new planets to explore.
    The goal is to terraform the final planet before the system collapses.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Increase prism size
    - actions[2]: Shift button (0=released, 1=held) -> Decrease prism size

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +10 for terraforming a planet.
    - +100 for terraforming the final planet (winning).
    - -100 for falling off a planet (losing).
    - +0.01 for a valid move.
    - -0.01 for an invalid move or resize action.
    - -0.001 per step to encourage speed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a prism through a shrinking planetary system. Match your prism's size to terraform planets and win before the system collapses."
    user_guide = "Use the arrow keys (↑↓←→) to move. Hold Space to increase the prism's size and Shift to decrease it."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    NUM_PLANETS = 5
    PRISM_MOVE_SPEED = 8
    PRISM_RESIZE_SPEED = 2
    PRISM_MIN_SIZE = 20
    PRISM_MAX_SIZE = 80

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_GLOW = (255, 200, 0, 50)
    COLOR_PLANET = (70, 130, 180)
    COLOR_PLANET_GLOW = (70, 130, 180, 50)
    COLOR_TERRAFORMED = (60, 220, 120)
    COLOR_TERRAFORMED_GLOW = (60, 220, 120, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 255, 255)
    COLOR_UI_SUCCESS = (60, 220, 120)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_puzzle = pygame.font.Font(None, 36)
        self.font_puzzle_letters = pygame.font.Font(None, 48)

        # --- Word Puzzle Data ---
        self.words = {
            20: "GO", 30: "RUN", 40: "JUMP", 50: "QUICK",
            60: "SYSTEM", 70: "EXPLORE", 80: "GALACTIC"
        }

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prism_pos = None
        self.prism_size = 0
        self.planets = []
        self.active_planet_idx = 0
        self.planet_shrink_rate = 0.0
        self.starfield = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_level()
        # FIX: pygame.math.Vector2 does not have a .copy() method.
        # Create a new vector from the existing one to copy it.
        self.prism_pos = pygame.math.Vector2(self.planets[0]['pos'])
        self.prism_size = self.PRISM_MIN_SIZE
        self.active_planet_idx = 0
        self.planet_shrink_rate = 0.05

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.001  # Time penalty

        # --- Unpack Actions ---
        movement, increase_size, decrease_size = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        reward += self._handle_resizing(increase_size, decrease_size)
        reward += self._handle_movement(movement)

        # --- Update Game State ---
        self._update_planets()
        terra_reward = self._check_terraforming()
        reward += terra_reward

        self.score += reward
        terminated = self._check_termination()

        # The environment does not use truncation, it's always False.
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.planets = []
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        
        # Create starfield using self.np_random for reproducibility
        self.starfield = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), 
             self.np_random.integers(0, self.SCREEN_HEIGHT), 
             self.np_random.integers(1, 3))
            for _ in range(150)
        ]

        # Create planets
        last_pos = pygame.Vector2(center_x, center_y)
        available_sizes = sorted(self.words.keys())

        for i in range(self.NUM_PLANETS):
            angle = self.np_random.uniform(0, 2 * math.pi)
            distance = self.np_random.uniform(100, 150) if i > 0 else 0
            pos = last_pos + pygame.Vector2(math.cos(angle) * distance, math.sin(angle) * distance)
            
            # Clamp position to be within screen bounds
            pos.x = np.clip(pos.x, 80, self.SCREEN_WIDTH - 80)
            pos.y = np.clip(pos.y, 80, self.SCREEN_HEIGHT - 80)
            
            radius = self.np_random.uniform(70, 90)
            target_size = self.np_random.choice(available_sizes)

            self.planets.append({
                'pos': pos,
                'initial_radius': radius,
                'radius': radius,
                'target_size': target_size,
                'terraformed': False,
                'active': (i == 0)
            })
            last_pos = pos

    def _handle_resizing(self, increase, decrease):
        reward = 0
        initial_size = self.prism_size
        if increase and not decrease:
            self.prism_size += self.PRISM_RESIZE_SPEED
        elif decrease and not increase:
            self.prism_size -= self.PRISM_RESIZE_SPEED

        self.prism_size = np.clip(self.prism_size, self.PRISM_MIN_SIZE, self.PRISM_MAX_SIZE)

        if self.prism_size == initial_size and (increase or decrease):
            reward -= 0.01  # Penalize trying to resize past limits
        return reward

    def _handle_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right

        if move_vec.length() > 0:
            new_pos = self.prism_pos + move_vec * self.PRISM_MOVE_SPEED
            if self._is_prism_supported(new_pos, self.prism_size):
                self.prism_pos = new_pos
                return 0.01  # Valid move
            else:
                return -0.01 # Invalid move
        return 0

    def _update_planets(self):
        # Increase shrink rate over time
        if self.steps > 0 and self.steps % 50 == 0:
            self.planet_shrink_rate += 0.01

        for planet in self.planets:
            if planet['active']:
                planet['radius'] = max(self.prism_size / 2, planet['radius'] - self.planet_shrink_rate)
    
    def _is_prism_supported(self, prism_pos, prism_size):
        prism_rect = pygame.Rect(
            prism_pos.x - prism_size / 2,
            prism_pos.y - prism_size / 2,
            prism_size,
            prism_size
        )
        for planet in self.planets:
            if planet['active']:
                # Check if all four corners of the prism are on the planet
                corners = [prism_rect.topleft, prism_rect.topright, prism_rect.bottomleft, prism_rect.bottomright]
                if all(planet['pos'].distance_to(corner) <= planet['radius'] for corner in corners):
                    return True
        return False

    def _check_terraforming(self):
        planet = self.planets[self.active_planet_idx]
        if not planet['terraformed'] and self.prism_size == planet['target_size']:
            if self._is_prism_supported(self.prism_pos, self.prism_size):
                planet['terraformed'] = True
                
                if self.active_planet_idx + 1 < self.NUM_PLANETS:
                    self.active_planet_idx += 1
                    self.planets[self.active_planet_idx]['active'] = True
                    return 10.0 # Reward for terraforming
                else:
                    # Final planet terraformed - Win condition
                    self.game_over = True
                    return 100.0
        return 0.0

    def _check_termination(self):
        if self.game_over: # Win condition already met
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        if not self._is_prism_supported(self.prism_pos, self.prism_size):
            self.game_over = True
            self.score -= 100 # Penalty for falling off
            return True
        
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Background Stars ---
        for x, y, size in self.starfield:
            # Twinkle effect
            if self.np_random.random() > 0.995:
                pygame.draw.circle(self.screen, self.COLOR_UI_ACCENT, (x, y), size)
            else:
                pygame.draw.circle(self.screen, self.COLOR_UI_TEXT, (x, y), size)

        # --- Draw Planets ---
        for planet in self.planets:
            if planet['active']:
                pos = (int(planet['pos'].x), int(planet['pos'].y))
                radius = int(planet['radius'])
                color = self.COLOR_TERRAFORMED if planet['terraformed'] else self.COLOR_PLANET
                glow_color = self.COLOR_TERRAFORMED_GLOW if planet['terraformed'] else self.COLOR_PLANET_GLOW
                
                # Draw glow
                glow_radius = int(radius * 1.2)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

                # Draw planet
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_UI_ACCENT)

        # --- Draw Prism ---
        prism_half = self.prism_size / 2
        prism_rect = pygame.Rect(
            int(self.prism_pos.x - prism_half),
            int(self.prism_pos.y - prism_half),
            int(self.prism_size),
            int(self.prism_size)
        )
        # Draw glow
        glow_rect = prism_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW, s.get_rect(), border_radius=3)
        self.screen.blit(s, glow_rect.topleft)
        # Draw prism
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, prism_rect, border_radius=2)

        # --- Draw UI ---
        self._render_ui()

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 30))
        
        planet_text = self.font_ui.render(f"PLANET: {self.active_planet_idx + 1}/{self.NUM_PLANETS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(planet_text, (self.SCREEN_WIDTH - planet_text.get_width() - 10, 10))

        # Puzzle / Target Size display
        if not self.game_over:
            current_planet = self.planets[self.active_planet_idx]
            target_size = current_planet['target_size']
            word = self.words.get(target_size, "?????")
            
            is_match = self.prism_size == target_size
            puzzle_color = self.COLOR_UI_SUCCESS if is_match else self.COLOR_UI_TEXT

            # Draw background panel
            panel_rect = pygame.Rect(0, 0, 400, 80)
            panel_rect.centerx = self.SCREEN_WIDTH // 2
            panel_rect.bottom = self.SCREEN_HEIGHT - 10
            panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
            panel_surface.fill((25, 30, 50, 180))
            self.screen.blit(panel_surface, panel_rect.topleft)

            # Draw title
            title_text = self.font_puzzle.render("TERRAFORM SEQUENCE", True, self.COLOR_UI_TEXT)
            title_rect = title_text.get_rect(centerx=panel_rect.width // 2, top=10)
            panel_surface.blit(title_text, title_rect)
            
            # Draw word letters
            letter_spacing = 40
            total_width = (len(word) - 1) * letter_spacing
            start_x = (panel_rect.width - total_width) // 2
            
            for i, letter in enumerate(word):
                letter_surf = self.font_puzzle_letters.render(letter, True, puzzle_color)
                letter_rect = letter_surf.get_rect(center=(start_x + i * letter_spacing, 55))
                panel_surface.blit(letter_surf, letter_rect)

            self.screen.blit(panel_surface, panel_rect.topleft)

        # Game Over message
        if self.game_over:
            is_win = self.planets[-1]['terraformed']
            message = "SYSTEM STABILIZED" if is_win else "SYSTEM COLLAPSED"
            color = self.COLOR_UI_SUCCESS if is_win else self.COLOR_PLAYER
            
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(s, (0, 0))

            end_text = self.font_puzzle_letters.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows for manual play and is not run by the tests.
    # It requires a display to be available.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Prism Terraformer")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print("\n--- Controls ---")
    print("Arrows: Move Prism")
    print("Space: Increase Size")
    print("Shift: Decrease Size")
    print("R: Reset Environment")
    print("----------------\n")

    while not done:
        # Action defaults
        movement = 0
        increase_size = 0
        decrease_size = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("Environment Reset.")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: increase_size = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: decrease_size = 1

        action = [movement, increase_size, decrease_size]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    env.close()