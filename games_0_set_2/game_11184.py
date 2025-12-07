import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:34:57.673123
# Source Brief: brief_01184.md
# Brief Index: 1184
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment for a visually-rich stealth/strategy game.
    The player acts as a quantum gardener, pollinating glowing flora while avoiding detection
    by roaming pests. The game prioritizes visual appeal and smooth gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Act as a quantum gardener, pollinating glowing flora while avoiding detection by roaming pests."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the selector and press space to start or stop pollination."
    )
    auto_advance = True

    # --- Persistent State Across Resets ---
    # As per the brief, score and unlocks persist between episodes.
    persistent_score = 0
    persistent_successful_pollinations = 0
    persistent_unlocked_plants = {0} # Set of unlocked plant type indices

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Core Gymnasium Setup ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame and Display ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Visual & Style Constants ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PEST = (255, 50, 80)
        self.COLOR_POLLINATION = (80, 150, 255)
        self.COLOR_WARNING = (255, 220, 50)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.font_main = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 40)
        self._create_background_nebula()

        # --- Game Mechanics Constants ---
        self.MAX_STEPS = 5000
        self.GRID_COLS, self.GRID_ROWS = 4, 3
        self.CELL_W = self.WIDTH // self.GRID_COLS
        self.CELL_H = self.HEIGHT // self.GRID_ROWS
        self.POLLINATION_DURATION = 150  # steps
        self.PEST_DETECTION_RADIUS = 50
        self.PEST_WARNING_RADIUS = 100

        self.PLANT_SPECS = [
            {'name': 'Quantum Orb', 'unlock_score': 0, 'color': (0, 255, 150)},
            {'name': 'Pulsar Triangle', 'unlock_score': 10, 'color': (150, 255, 0)},
            {'name': 'Stellar Bloom', 'unlock_score': 50, 'color': (255, 180, 0)},
            {'name': 'Nebula Spire', 'unlock_score': 100, 'color': (200, 100, 255)},
        ]

        # --- Initialize State Variables ---
        self.steps = 0
        self.selector_pos = [0, 0]
        self.plants = []
        self.pests = []
        self.particles = []
        self.pollinating_plant_idx = None
        self.pollination_progress = 0
        self.last_space_held = False
        self.last_reward_info = {}

        # --- Final Setup ---
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.selector_pos = [0, 0]
        self.last_space_held = False
        self.pollinating_plant_idx = None
        self.pollination_progress = 0
        self.particles.clear()
        self.last_reward_info = {}

        self._initialize_plants()
        self._initialize_pests()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        reward = 0
        self.last_reward_info = {}

        # 1. Handle Player Input
        self._handle_input(movement, space_press)

        # 2. Update Game State
        self.steps += 1
        self._update_pests()
        self._update_particles()
        
        # 3. Update Pollination & Calculate Rewards
        reward += self._update_pollination()

        # 4. Check for Termination
        terminated = self.steps >= self.MAX_STEPS
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_press):
        prev_selector_pos = list(self.selector_pos)
        if movement == 1: self.selector_pos[1] = (self.selector_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3: self.selector_pos[0] = (self.selector_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_COLS

        selected_idx = self.selector_pos[1] * self.GRID_COLS + self.selector_pos[0]

        # Moving selector cancels pollination
        if self.selector_pos != prev_selector_pos and self.pollinating_plant_idx is not None:
            self.pollinating_plant_idx = None
            self.pollination_progress = 0
            # sfx: cancel_sound

        if space_press:
            if self.pollinating_plant_idx == selected_idx:
                # Cancel current pollination
                self.pollinating_plant_idx = None
                self.pollination_progress = 0
                # sfx: cancel_sound
            else:
                # Start new pollination
                self.pollinating_plant_idx = selected_idx
                self.pollination_progress = 0
                # sfx: start_pollination_sound

    def _update_pests(self):
        base_speed = 0.005 + 0.05 * (GameEnv.persistent_successful_pollinations // 500)
        for pest in self.pests:
            pest['angle'] += pest['speed_mult'] * base_speed
            pest['pos'] = (
                pest['center'][0] + pest['radius'] * math.cos(pest['angle']),
                pest['center'][1] + pest['radius'] * math.sin(pest['angle'])
            )

    def _update_pollination(self):
        reward = 0
        if self.pollinating_plant_idx is None:
            return 0

        plant = self.plants[self.pollinating_plant_idx]
        is_detected = False
        is_warned = False

        # Check for pest proximity
        for pest in self.pests:
            dist = math.hypot(plant['center_pos'][0] - pest['pos'][0], plant['center_pos'][1] - pest['pos'][1])
            if dist < self.PEST_DETECTION_RADIUS:
                is_detected = True
                break
            if dist < self.PEST_WARNING_RADIUS:
                is_warned = True
        
        plant['is_warned'] = is_warned

        if is_detected:
            # sfx: detection_fail_sound
            self.pollinating_plant_idx = None
            self.pollination_progress = 0
            self.last_reward_info = {'event': 'detection', 'value': -5}
            return -5

        # Progress pollination
        self.pollination_progress += 1
        reward += 0.1  # Continuous reward for safe pollination
        self.last_reward_info = {'event': 'pollinating', 'value': 0.1}

        # Create pollination particles
        if self.steps % 2 == 0:
            self._create_particles(plant['center_pos'], 1, self.COLOR_POLLINATION)

        # Check for completion
        if self.pollination_progress >= self.POLLINATION_DURATION:
            # sfx: success_sound
            GameEnv.persistent_score += 1
            GameEnv.persistent_successful_pollinations += 1
            reward += 1
            self.last_reward_info = {'event': 'complete', 'value': 1}

            # Check for unlocks
            newly_unlocked = False
            for i, spec in enumerate(self.PLANT_SPECS):
                if i not in GameEnv.persistent_unlocked_plants and GameEnv.persistent_score >= spec['unlock_score']:
                    GameEnv.persistent_unlocked_plants.add(i)
                    newly_unlocked = True
                    # sfx: unlock_sound
            
            if newly_unlocked:
                reward += 10
                self.last_reward_info = {'event': 'unlock', 'value': 10}
                self._initialize_plants() # Re-roll plants with new types available

            self.pollinating_plant_idx = None
            self.pollination_progress = 0
        
        return reward

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": GameEnv.persistent_score,
            "steps": self.steps,
            "pollinations": GameEnv.persistent_successful_pollinations,
            "unlocked_plants": len(GameEnv.persistent_unlocked_plants),
            "last_reward_info": self.last_reward_info
        }

    def _render_game(self):
        self._draw_grid()
        for i, plant in enumerate(self.plants):
            self._draw_plant(plant, is_selected=(i == self.selector_pos[1] * self.GRID_COLS + self.selector_pos[0]))
        self._draw_particles()
        for pest in self.pests:
            self._draw_glowing_circle(self.screen, pest['pos'], 12, self.COLOR_PEST, 3)
        self._draw_selector()

    def _render_ui(self):
        # Score Display
        score_text = self.font_score.render(f"SCORE: {GameEnv.persistent_score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Pollination Progress Bar
        if self.pollinating_plant_idx is not None:
            plant = self.plants[self.pollinating_plant_idx]
            bar_w = 100
            bar_h = 10
            bar_x = plant['center_pos'][0] - bar_w / 2
            bar_y = plant['center_pos'][1] - 60
            
            progress = self.pollination_progress / self.POLLINATION_DURATION
            
            pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h), 1)
            pygame.draw.rect(self.screen, self.COLOR_POLLINATION, (bar_x, bar_y, bar_w * progress, bar_h))

        # Unlocked Plants Display
        icon_size = 20
        padding = 10
        start_x = self.WIDTH - padding - (icon_size + padding) * len(self.PLANT_SPECS)
        for i, spec in enumerate(self.PLANT_SPECS):
            x = start_x + i * (icon_size + padding)
            y = self.HEIGHT - padding - icon_size
            rect = (x, y, icon_size, icon_size)
            if i in GameEnv.persistent_unlocked_plants:
                pygame.draw.rect(self.screen, spec['color'], rect, border_radius=3)
            else:
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=3)


    def _initialize_plants(self):
        self.plants.clear()
        available_types = sorted(list(GameEnv.persistent_unlocked_plants))
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                center_x = c * self.CELL_W + self.CELL_W / 2
                center_y = r * self.CELL_H + self.CELL_H / 2
                plant_type = self.np_random.choice(available_types)
                self.plants.append({
                    'pos': (c, r),
                    'center_pos': (center_x, center_y),
                    'type': plant_type,
                    'growth': self.np_random.uniform(0.8, 1.2),
                    'phase': self.np_random.uniform(0, 2 * math.pi),
                    'is_warned': False
                })

    def _initialize_pests(self):
        self.pests.clear()
        num_pests = 3
        for i in range(num_pests):
            self.pests.append({
                'center': (self.np_random.uniform(self.WIDTH*0.2, self.WIDTH*0.8), 
                           self.np_random.uniform(self.HEIGHT*0.2, self.HEIGHT*0.8)),
                'radius': self.np_random.uniform(80, 180),
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'speed_mult': self.np_random.choice([-1, 1]) * self.np_random.uniform(0.8, 1.2),
                'pos': (0, 0)
            })

    def _create_background_nebula(self):
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.bg_surface.fill(self.COLOR_BG)
        for _ in range(150):
            pos = (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
            radius = random.randint(20, 100)
            color = (
                random.randint(15, 30),
                random.randint(10, 25),
                random.randint(20, 40),
                random.randint(10, 30) # Alpha
            )
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.bg_surface.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _draw_grid(self):
        for c in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_W, 0), (c * self.CELL_W, self.HEIGHT), 1)
        for r in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_H), (self.WIDTH, r * self.CELL_H), 1)

    def _draw_selector(self):
        c, r = self.selector_pos
        rect = (c * self.CELL_W + 5, r * self.CELL_H + 5, self.CELL_W - 10, self.CELL_H - 10)
        
        # Pulsating alpha for the selector
        alpha = 150 + 100 * math.sin(self.steps * 0.1)
        color = (*self.COLOR_SELECTOR, alpha)
        
        temp_surf = pygame.Surface((self.CELL_W - 10, self.CELL_H - 10), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), 2, border_radius=8)
        self.screen.blit(temp_surf, (rect[0], rect[1]))

    def _draw_plant(self, plant, is_selected):
        spec = self.PLANT_SPECS[plant['type']]
        color = spec['color']
        pos = plant['center_pos']
        pulse = 0.5 + 0.5 * math.sin(self.steps * 0.05 + plant['phase'])
        
        if plant['is_warned']:
            warn_alpha = 100 + 100 * math.sin(self.steps * 0.2)
            self._draw_glowing_circle(self.screen, pos, self.PEST_WARNING_RADIUS, (*self.COLOR_WARNING, warn_alpha), 1, is_ring=True)

        if plant['type'] == 0: # Quantum Orb
            radius = 15 * plant['growth'] * (1 + 0.1 * pulse)
            self._draw_glowing_circle(self.screen, pos, radius, color, 4)
        elif plant['type'] == 1: # Pulsar Triangle
            size = 20 * plant['growth']
            angle = self.steps * 0.02 + plant['phase']
            points = []
            for i in range(3):
                a = angle + i * 2 * math.pi / 3
                points.append((pos[0] + size * math.cos(a), pos[1] + size * math.sin(a)))
            self._draw_glowing_polygon(self.screen, points, color, 3)
        elif plant['type'] == 2: # Stellar Bloom
            base_radius = 20 * plant['growth']
            num_petals = 5
            for i in range(num_petals):
                a = self.steps * 0.01 + plant['phase'] + i * 2 * math.pi / num_petals
                petal_pos = (pos[0] + base_radius * math.cos(a), pos[1] + base_radius * math.sin(a))
                petal_radius = 8 * plant['growth'] * (1 + 0.2 * pulse)
                self._draw_glowing_circle(self.screen, petal_pos, petal_radius, color, 2)
            self._draw_glowing_circle(self.screen, pos, 8 * plant['growth'], (255, 255, 200), 2)
        elif plant['type'] == 3: # Nebula Spire
            height = 35 * plant['growth'] * (1 + 0.1 * pulse)
            points = [
                (pos[0] - 8, pos[1] + 15),
                (pos[0] + 8, pos[1] + 15),
                (pos[0], pos[1] - height)
            ]
            self._draw_glowing_polygon(self.screen, points, color, 3)

    def _draw_glowing_circle(self, surface, pos, radius, color, layers=5, is_ring=False):
        pos_int = (int(pos[0]), int(pos[1]))
        max_radius = int(radius * (1.8 if not is_ring else 1.2))
        
        temp_surf = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
        center = (max_radius, max_radius)

        for i in range(layers, 0, -1):
            alpha = int(color[3] if len(color) == 4 else 255 * (1 / (layers - i + 2)))
            layer_radius = int(radius * (1 + i / layers * (0.8 if not is_ring else 0.2)))
            layer_color = (color[0], color[1], color[2], alpha)
            if is_ring:
                pygame.gfxdraw.aacircle(temp_surf, center[0], center[1], layer_radius, layer_color)
            else:
                pygame.gfxdraw.filled_circle(temp_surf, center[0], center[1], layer_radius, layer_color)
        
        surface.blit(temp_surf, (pos_int[0] - max_radius, pos_int[1] - max_radius))

    def _draw_glowing_polygon(self, surface, points, color, layers=3):
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        width = int(max_x - min_x) + 20
        height = int(max_y - min_y) + 20
        
        temp_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        rel_points = [(p[0] - min_x + 10, p[1] - min_y + 10) for p in points]

        for i in range(layers, 0, -1):
            alpha = int(255 * (1 / (layers - i + 2)))
            layer_color = (*color, alpha)
            
            # This is a simplified glow for polygons, drawing it multiple times isn't great
            # but it's a decent approximation without complex shaders.
            pygame.gfxdraw.filled_polygon(temp_surf, rel_points, layer_color)
            pygame.gfxdraw.aapolygon(temp_surf, rel_points, layer_color)

        surface.blit(temp_surf, (min_x - 10, min_y - 10))

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(30, 60),
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98
            p['vel'][1] *= 0.98
            p['life'] -= 1

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 60))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / 15)
            pygame.draw.circle(self.screen, color, pos, max(1, radius))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run this, you might need to uninstall the dummy driver or set the environment variable differently
    # For example: `SDL_VIDEODRIVER=x11 python your_script.py`
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Override the screen for direct display
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Quantum Garden")

    done = False
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print("Arrows: Move selector")
    print("Space: Start/Stop pollination")
    print("Q: Quit")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                # Map keys to actions
                if event.key == pygame.K_UP: movement = 1
                if event.key == pygame.K_DOWN: movement = 2
                if event.key == pygame.K_LEFT: movement = 3
                if event.key == pygame.K_RIGHT: movement = 4
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Info: {info['last_reward_info']}, Score: {info['score']}")

        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()