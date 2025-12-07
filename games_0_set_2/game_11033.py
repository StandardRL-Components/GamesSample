import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:22:54.092152
# Source Brief: brief_01033.md
# Brief Index: 1033
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A Gymnasium environment where the agent manages a planet's ecosystem.
    The agent plays element cards onto a grid-based planet to maintain balance.
    Complex interactions between elements create a dynamic puzzle. The goal is to
    achieve a high stability score by carefully considering the cascading effects
    of each placement. New, more volatile elements are unlocked as the score increases,
    ramping up the difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a planet's ecosystem by playing element cards. Balance the elements to maintain stability and unlock new powers."
    )
    user_guide = (
        "Use number keys (1-4) to select an element card. Press space to place it in the center, or hold shift and press space to place it randomly."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLANET_RADIUS = 150
    PLANET_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)
    GRID_SIZE = 16
    MAX_STEPS = 5000
    
    # --- Colors ---
    COLOR_BG = (15, 20, 45)
    COLOR_PLANET_BORDER = (80, 90, 120)
    COLOR_GRID = (255, 255, 255, 20)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_HIGHLIGHT = (255, 255, 100)
    
    # --- Element Definitions ---
    ELEMENTS = {
        0: {"name": "FIRE", "color": (255, 80, 20)},
        1: {"name": "WATER", "color": (50, 150, 255)},
        2: {"name": "EARTH", "color": (120, 200, 80)},
        3: {"name": "AIR", "color": (240, 240, 150)},
        4: {"name": "LIGHT", "color": (255, 255, 255)},
        5: {"name": "DARK", "color": (80, 40, 120)},
    }
    NUM_ELEMENTS = len(ELEMENTS)

    # --- Interaction Matrix: How element row affects element col ---
    # Stronger reactions for later-unlocked elements
    INTERACTION_MATRIX = np.array([
        # FIRE, WATER, EARTH, AIR, LIGHT, DARK
        [ 0.0, -1.0,  0.5,  0.2,  0.1, -0.2], # FIRE
        [-1.0,  0.0,  0.2,  0.5, -0.2,  0.1], # WATER
        [-0.5,  0.2,  0.0, -1.0,  0.5, -0.1], # EARTH
        [-0.2, -0.5, -1.0,  0.0,  0.1,  0.5], # AIR
        [ 0.1, -0.2, -0.5,  0.1,  0.0, -1.5], # LIGHT
        [-0.2,  0.1,  0.5, -0.1, -1.5,  0.0], # DARK
    ]) * 0.1 # Global interaction strength scaler

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 16)
        
        # State variables initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.planet_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE, self.NUM_ELEMENTS), dtype=np.float32)
        self.inventory = []
        self.unlocked_elements = []
        self.selected_card_idx = -1
        self.particles = []
        self.last_stability = 0.0
        self.unlock_thresholds = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.planet_grid.fill(0)
        
        self.unlocked_elements = list(range(4)) # Start with Fire, Water, Earth, Air
        self.unlock_thresholds = {
            4: 500,  # Light unlocks at 500 score
            5: 1000  # Dark unlocks at 1000 score
        }
        
        self.inventory = [self._draw_new_card() for _ in range(4)]
        self.selected_card_idx = -1
        self.particles = []
        self.last_stability = self._calculate_stability()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        turn_taken = False
        reward = 0.0
        
        old_score = self.score
        
        # 1. Handle card selection
        if 1 <= movement <= 4:
            self.selected_card_idx = movement - 1
        
        # 2. Handle card placement or skipping a turn
        if space_held and self.selected_card_idx != -1:
            # Place card
            card_element_idx = self.inventory[self.selected_card_idx]
            
            if shift_held: # Place randomly
                px, py = self.np_random.integers(0, self.GRID_SIZE, size=2)
            else: # Place in center
                px, py = self.GRID_SIZE // 2, self.GRID_SIZE // 2
            
            self._apply_element(card_element_idx, (px, py))
            # SFX: Card placement sound
            
            self.inventory[self.selected_card_idx] = self._draw_new_card()
            self.selected_card_idx = -1 # Deselect after use
            turn_taken = True
        elif movement == 0: # No-op is a valid turn
            turn_taken = True
            # SFX: Skip turn sound

        if turn_taken:
            self.steps += 1
            self._update_planet_state()
            
            # --- Calculate Reward ---
            new_stability = self._calculate_stability()
            self.score += new_stability * 5 # Stability contributes to score
            
            # Reward for improving stability
            if new_stability > self.last_stability:
                reward += 1.0
            elif new_stability < self.last_stability:
                reward -= 1.0
            self.last_stability = new_stability
            
            # Reward for score milestones
            if math.floor(self.score / 100) > math.floor(old_score / 100):
                reward += 10.0
                # SFX: Milestone achievement sound
            
            # Check for and reward element unlocks
            unlocked_something = self._check_unlocks()
            if unlocked_something:
                reward += 5.0
                # SFX: Unlock sound effect
        
        terminated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _apply_element(self, element_idx, pos):
        px, py = pos
        # Add a burst of the new element
        self.planet_grid[px, py, element_idx] += 2.0
        
        # Spawn particles for visual feedback
        element_color = self.ELEMENTS[element_idx]["color"]
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(20, 40)
            grid_center_x = self.PLANET_CENTER[0] + (px - self.GRID_SIZE / 2 + 0.5) * (2 * self.PLANET_RADIUS / self.GRID_SIZE)
            grid_center_y = self.PLANET_CENTER[1] + (py - self.GRID_SIZE / 2 + 0.5) * (2 * self.PLANET_RADIUS / self.GRID_SIZE)
            self.particles.append([
                [grid_center_x, grid_center_y], # pos
                list(vel), # vel
                element_color,
                lifespan
            ])

    def _update_planet_state(self):
        # 1. Interactions
        new_grid = self.planet_grid.copy()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_values = self.planet_grid[r, c]
                if np.sum(cell_values) > 0.1: # Only process active cells
                    deltas = np.zeros(self.NUM_ELEMENTS)
                    for i in range(self.NUM_ELEMENTS):
                        for j in range(self.NUM_ELEMENTS):
                            if i != j:
                                reaction_rate = self.INTERACTION_MATRIX[i, j]
                                deltas[j] += cell_values[i] * reaction_rate * 0.1 # Scaled update
                    new_grid[r, c] += deltas
        
        # 2. Diffusion
        diffused_grid = new_grid.copy()
        diffusion_rate = 0.05
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                for i in range(self.GRID_SIZE):
                    for j in range(self.GRID_SIZE):
                        dist_sq = (r-i)**2 + (c-j)**2
                        if dist_sq == 1: # Neighbors
                           diffused_grid[i, j] += new_grid[r, c] * diffusion_rate
                           diffused_grid[r, c] -= new_grid[r, c] * diffusion_rate
        
        self.planet_grid = np.clip(diffused_grid, 0, 10) # Clamp values

    def _calculate_stability(self):
        total_elements = np.sum(self.planet_grid)
        if total_elements < 1e-6:
            return 1.0 # Perfect stability if empty
        
        # Stability is higher when elements are balanced
        # We use inverse of standard deviation of element totals
        element_sums = np.sum(self.planet_grid, axis=(0, 1))
        std_dev = np.std(element_sums[element_sums > 0]) # Only consider present elements
        
        return 1.0 / (1.0 + std_dev)

    def _check_unlocks(self):
        unlocked_new = False
        elements_to_unlock = sorted(self.unlock_thresholds.keys())
        for element_idx in elements_to_unlock:
            if element_idx not in self.unlocked_elements and self.score >= self.unlock_thresholds[element_idx]:
                self.unlocked_elements.append(element_idx)
                unlocked_new = True
        return unlocked_new

    def _draw_new_card(self):
        return self.np_random.choice(self.unlocked_elements)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render planet background color based on element balance
        total_elements = np.sum(self.planet_grid)
        if total_elements > 1e-6:
            element_sums = np.sum(self.planet_grid, axis=(0, 1))
            avg_color = np.zeros(3)
            for i in range(self.NUM_ELEMENTS):
                avg_color += np.array(self.ELEMENTS[i]["color"]) * element_sums[i]
            avg_color /= total_elements
            planet_bg_color = tuple(np.clip(avg_color, 0, 255).astype(int))
        else:
            planet_bg_color = (50, 55, 80)
        
        pygame.gfxdraw.filled_circle(self.screen, self.PLANET_CENTER[0], self.PLANET_CENTER[1], self.PLANET_RADIUS, planet_bg_color)

        # Render planet grid cells
        cell_size = (2 * self.PLANET_RADIUS) / self.GRID_SIZE
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_sum = np.sum(self.planet_grid[r,c])
                if cell_sum > 0.1:
                    cell_color = np.zeros(3)
                    for i in range(self.NUM_ELEMENTS):
                        cell_color += np.array(self.ELEMENTS[i]["color"]) * self.planet_grid[r, c, i]
                    cell_color /= cell_sum
                    
                    alpha = min(255, int(cell_sum * 25))
                    color_with_alpha = (*tuple(np.clip(cell_color, 0, 255).astype(int)), alpha)
                    
                    x = self.PLANET_CENTER[0] - self.PLANET_RADIUS + c * cell_size
                    y = self.PLANET_CENTER[1] - self.PLANET_RADIUS + r * cell_size
                    cell_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                    cell_surf.fill(color_with_alpha)
                    self.screen.blit(cell_surf, (int(x), int(y)))

        pygame.gfxdraw.aacircle(self.screen, self.PLANET_CENTER[0], self.PLANET_CENTER[1], self.PLANET_RADIUS, self.COLOR_PLANET_BORDER)
        
        # Render particles
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[3] -= 1
            if p[3] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[3] / 40))
                color = (*p[2], alpha)
                surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (2,2), 2)
                self.screen.blit(surf, (int(p[0][0]-2), int(p[0][1]-2)))

    def _render_ui(self):
        # Render Score and Turn
        score_text = f"SCORE: {int(self.score)}"
        turn_text = f"TURN: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(score_text, (self.SCREEN_WIDTH - 10, 10), self.font_main, self.COLOR_TEXT, align="topright")
        self._draw_text(turn_text, (self.SCREEN_WIDTH - 10, 40), self.font_small, self.COLOR_TEXT, align="topright")
        
        # Render Inventory
        card_size = 50
        padding = 10
        start_x = self.SCREEN_WIDTH // 2 - (4 * card_size + 3 * padding) // 2
        y = self.SCREEN_HEIGHT - card_size - 10
        
        for i in range(4):
            card_x = start_x + i * (card_size + padding)
            rect = pygame.Rect(card_x, y, card_size, card_size)
            
            element_idx = self.inventory[i]
            color = self.ELEMENTS[element_idx]["color"]
            
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*0.5 for c in color), rect, width=2, border_radius=5)
            
            if i == self.selected_card_idx:
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, width=3, border_radius=5)

    def _draw_text(self, text, pos, font, color, align="topleft"):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        main = font.render(text, True, color)
        
        text_rect = main.get_rect()
        shadow_rect = shadow.get_rect()

        if align == "topright":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        shadow_rect.topleft = (text_rect.left + 2, text_rect.top + 2)

        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(main, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stability": self.last_stability,
            "unlocked_elements": len(self.unlocked_elements)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This __main__ block is for manual play and is not part of the environment's core API.
    # It will not be executed by the automated test suite.
    try:
        env = GameEnv()
        
        # --- Manual Play ---
        # Un-set the dummy driver to allow for display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        pygame.display.set_caption("Eco-Strategy Planet")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        done = False
        
        action = [0, 0, 0] # [movement, space, shift]
        
        print("--- Controls ---")
        print(GameEnv.user_guide)
        print("R: Reset environment")
        print("Q: Quit")
        
        while not done:
            action = [0, 0, 0] # Reset action every frame
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                    
                    # Map keys to actions for one step
                    if event.key == pygame.K_1: action[0] = 1
                    elif event.key == pygame.K_2: action[0] = 2
                    elif event.key == pygame.K_3: action[0] = 3
                    elif event.key == pygame.K_4: action[0] = 4
                    elif event.key == pygame.K_n: action[0] = 0
                    
                    if event.key == pygame.K_SPACE: action[1] = 1
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            # Only step if a turn-advancing action is taken
            # A turn is selecting a card (1-4), placing a card (space), or explicitly passing (n)
            is_turn_action = (action[0] != 0 or action[1] == 1) or (event.type == pygame.KEYDOWN and event.key == pygame.K_n)
            
            # The original code's manual play logic was a bit complex.
            # For `auto_advance=False`, we step only on a meaningful action.
            # Let's simplify: any keydown that results in an action triggers a step.
            if event.type == pygame.KEYDOWN:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Stability: {info['stability']:.3f}")
                if terminated or truncated:
                    print("Episode finished.")
                    obs, info = env.reset()

            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit to 30 FPS

    finally:
        if 'env' in locals():
            env.close()