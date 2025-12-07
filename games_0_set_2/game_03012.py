
# Generated: 2025-08-27T22:05:59.342312
# Source Brief: brief_03012.md
# Brief Index: 3012

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selector. Space to click the selected area."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced idle clicker. Click on colored areas to score points. "
        "Red areas are high-risk, high-reward. Reach 1000 points before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.GAME_DURATION_SECONDS
        self.WIN_SCORE = 1000
        self.AREA_UPDATE_RATE = 15 # Update area values every 15 frames

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_area = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_LOW_RISK = (0, 180, 120)
        self.COLOR_MEDIUM_RISK = (220, 180, 0)
        self.COLOR_HIGH_RISK = (210, 50, 50)
        self.COLOR_PARTICLE = [(255, 200, 80), (255, 160, 50), (255, 100, 30)]

        # --- State Variables ---
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.selected_index = 0
        self.space_was_held = False
        self.click_areas = []
        self.particles = []
        self.click_effects = []
        self.rng = None
        self.last_score_change = 0

        # --- Navigation Map ---
        # Defines how the selector moves between the 5 click areas
        self.nav_map = {
            # Center
            0: {1: 1, 2: 2, 3: 3, 4: 4},
            # Up
            1: {1: 1, 2: 0, 3: 1, 4: 1},
            # Down
            2: {1: 0, 2: 2, 3: 2, 4: 2},
            # Left
            3: {1: 3, 2: 3, 3: 3, 4: 0},
            # Right
            4: {1: 4, 2: 4, 3: 0, 4: 4}
        }
        
        # --- Initialize state ---
        self.reset()
        self.validate_implementation()

    def _setup_click_areas(self):
        self.click_areas = []
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        size = 100
        offset = 120

        # Area 0: Center (Low Risk)
        self.click_areas.append({
            'rect': pygame.Rect(cx - size // 2, cy - size // 2, size, size),
            'color': self.COLOR_LOW_RISK,
            'value_range': (5, 15), 'multiplier_range': (1.0, 1.2),
            'value': 10, 'multiplier': 1.0
        })
        # Area 1: Up (Medium Risk)
        self.click_areas.append({
            'rect': pygame.Rect(cx - size // 2, cy - offset - size // 2, size, size),
            'color': self.COLOR_MEDIUM_RISK,
            'value_range': (10, 30), 'multiplier_range': (0.8, 1.5),
            'value': 20, 'multiplier': 1.0
        })
        # Area 2: Down (Medium Risk)
        self.click_areas.append({
            'rect': pygame.Rect(cx - size // 2, cy + offset - size // 2, size, size),
            'color': self.COLOR_MEDIUM_RISK,
            'value_range': (10, 30), 'multiplier_range': (0.8, 1.5),
            'value': 20, 'multiplier': 1.0
        })
        # Area 3: Left (High Risk)
        self.click_areas.append({
            'rect': pygame.Rect(cx - offset - size // 2, cy - size // 2, size, size),
            'color': self.COLOR_HIGH_RISK,
            'value_range': (-20, 100), 'multiplier_range': (0.5, 3.0),
            'value': 40, 'multiplier': 1.0
        })
        # Area 4: Right (High Risk)
        self.click_areas.append({
            'rect': pygame.Rect(cx + offset - size // 2, cy - size // 2, size, size),
            'color': self.COLOR_HIGH_RISK,
            'value_range': (-20, 100), 'multiplier_range': (0.5, 3.0),
            'value': 40, 'multiplier': 1.0
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Create a default RNG if none is provided
            if self.rng is None:
                self.rng = np.random.default_rng()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_index = 0
        self.space_was_held = False
        self.particles.clear()
        self.click_effects.clear()
        self.last_score_change = 0

        self._setup_click_areas()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        self.last_score_change = 0 # Reset score change for this step

        if not self.game_over:
            self._update_game_state(action)
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Large reward for winning
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Selector Movement ---
        if movement != 0:
            self.selected_index = self.nav_map[self.selected_index][movement]

        # --- Handle Click Action ---
        is_click_event = space_held and not self.space_was_held
        if is_click_event:
            # sound_effect: "click.wav"
            area = self.click_areas[self.selected_index]
            points_gained = int(area['value'] * area['multiplier'])
            self.score += points_gained
            self.last_score_change = points_gained
            self._create_particles(area['rect'].center, points_gained)
            self._create_click_effect(area['rect'].center)
        self.space_was_held = space_held

        # --- Update Area Values Periodically ---
        if self.steps % self.AREA_UPDATE_RATE == 0:
            for area in self.click_areas:
                area['value'] = self.rng.integers(area['value_range'][0], area['value_range'][1] + 1)
                area['multiplier'] = self.rng.uniform(area['multiplier_range'][0], area['multiplier_range'][1])

        # --- Update Animations ---
        self._update_particles()
        self._update_click_effects()

    def _calculate_reward(self):
        return self.last_score_change * 0.1

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Render click areas
        for i, area in enumerate(self.click_areas):
            # Draw main area
            pygame.draw.rect(self.screen, area['color'], area['rect'], border_radius=10)
            
            # Draw selector highlight
            if i == self.selected_index:
                pygame.draw.rect(self.screen, self.COLOR_SELECTOR, area['rect'], 4, border_radius=10)
            
            # Draw text inside area
            potential_score = int(area['value'] * area['multiplier'])
            score_color = (255, 255, 255) if potential_score >= 0 else (255, 100, 100)
            score_text = self.font_area.render(f"{potential_score:+}", True, score_color)
            text_rect = score_text.get_rect(center=area['rect'].center)
            self.screen.blit(score_text, text_rect)

        # Render click effects and particles
        self._render_click_effects()
        self._render_particles()

    def _render_ui(self):
        # Score display
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Timer bar
        time_ratio = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
        time_bar_width = int((self.WIDTH / 3) * time_ratio)
        time_bar_rect = pygame.Rect(self.WIDTH - (self.WIDTH / 3) - 15, 15, time_bar_width, 20)
        
        # Animate color from green to red
        bar_color = (
            int(255 * (1 - time_ratio)),
            int(200 * time_ratio),
            50
        )
        pygame.draw.rect(self.screen, bar_color, time_bar_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - (self.WIDTH / 3) - 15, 15, self.WIDTH / 3, 20), 2, border_radius=5)
        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            result_surf = self.font_main.render(result_text, True, self.COLOR_SELECTOR)
            result_rect = result_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(result_surf, result_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Particle and Effect Helpers ---
    def _create_particles(self, pos, points):
        num_particles = min(50, max(5, abs(points) // 2))
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.rng.uniform(2, 5),
                'life': self.rng.integers(20, 40),
                'color': random.choice(self.COLOR_PARTICLE)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['radius'] -= 0.05
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color']
            )

    def _create_click_effect(self, pos):
        self.click_effects.append({
            'pos': pos,
            'radius': 10,
            'max_radius': 60,
            'life': 15,
            'max_life': 15
        })

    def _update_click_effects(self):
        for e in self.click_effects[:]:
            e['life'] -= 1
            e['radius'] += (e['max_radius'] - 10) / e['max_life']
            if e['life'] <= 0:
                self.click_effects.remove(e)

    def _render_click_effects(self):
        for e in self.click_effects:
            alpha = int(255 * (e['life'] / e['max_life']))
            color = (*self.COLOR_SELECTOR, alpha)
            pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius'])-1, color)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human interaction
    pygame.display.set_caption("Idle Clicker Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Human ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to blit it to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()