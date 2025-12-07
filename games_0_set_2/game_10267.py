import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:12:53.024917
# Source Brief: brief_00267.md
# Brief Index: 267
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls magnetic fields to guide
    falling metallic shapes into their matching slots. The game prioritizes
    visual quality and a satisfying "game feel" with smooth physics, particle
    effects, and clear UI.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide falling metallic shapes into their matching slots using a controllable magnetic field. "
        "Attract or repel shapes to solve the puzzle before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the magnet. Hold SPACE to attract shapes and SHIFT to repel them."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 90 * FPS  # 90 seconds

    # Colors
    COLOR_BG = (15, 18, 36)
    COLOR_GRID = (30, 35, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_MAGNET_ATTRACT = (100, 200, 255)
    COLOR_MAGNET_REPEL = (255, 150, 80)
    
    SHAPE_TYPES = ['circle', 'square', 'triangle', 'diamond', 'pentagon']
    SHAPE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255)   # Purple
    ]

    # Physics
    GRAVITY = 0.03
    MAGNET_FORCE_SCALE = 800.0
    PLAYER_SPEED = 4.0
    SHAPE_DAMPING = 0.995

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)


        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        
        self.player_magnet = None
        self.shapes = []
        self.slots = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        # Player Magnet
        self.player_magnet = {
            'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            'strength': 0, # -1 for repel, 0 for off, 1 for attract
            'radius': 15
        }
        
        self.particles = []
        
        # Create Slots
        self.slots = []
        num_slots = len(self.SHAPE_TYPES)
        slot_y = self.HEIGHT - 40
        slot_spacing = self.WIDTH / (num_slots + 1)
        slot_indices = list(range(num_slots))
        self.np_random.shuffle(slot_indices)

        for i in range(num_slots):
            idx = slot_indices[i]
            self.slots.append({
                'pos': pygame.Vector2(slot_spacing * (i + 1), slot_y),
                'type': self.SHAPE_TYPES[idx],
                'color': self.SHAPE_COLORS[idx],
                'radius': 25,
                'occupied_by': -1 # shape index
            })
            
        # Create Shapes
        self.shapes = []
        for i in range(num_slots):
            start_x = self.np_random.uniform(50, self.WIDTH - 50)
            start_y = self.np_random.uniform(20, 80)
            target_slot_idx = -1
            for j, slot in enumerate(self.slots):
                if slot['type'] == self.SHAPE_TYPES[i]:
                    target_slot_idx = j
                    break

            self.shapes.append({
                'pos': pygame.Vector2(start_x, start_y),
                'vel': pygame.Vector2(0, 0),
                'type': self.SHAPE_TYPES[i],
                'color': self.SHAPE_COLORS[i],
                'radius': 15,
                'is_placed': False,
                'target_slot_idx': target_slot_idx,
                'prev_dist_to_slot': pygame.Vector2(start_x, start_y).distance_to(self.slots[target_slot_idx]['pos'])
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.game_over = False

        # --- Update Player Magnet ---
        if movement == 1: self.player_magnet['pos'].y -= self.PLAYER_SPEED
        if movement == 2: self.player_magnet['pos'].y += self.PLAYER_SPEED
        if movement == 3: self.player_magnet['pos'].x -= self.PLAYER_SPEED
        if movement == 4: self.player_magnet['pos'].x += self.PLAYER_SPEED
        
        self.player_magnet['pos'].x = np.clip(self.player_magnet['pos'].x, 0, self.WIDTH)
        self.player_magnet['pos'].y = np.clip(self.player_magnet['pos'].y, 0, self.HEIGHT)
        
        self.player_magnet['strength'] = 0
        if space_held: self.player_magnet['strength'] = 1   # Attract
        if shift_held: self.player_magnet['strength'] = -1  # Repel
        # If both are held, repel wins
        
        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1
        
        for i, shape in enumerate(self.shapes):
            if shape['is_placed']:
                continue

            # Apply gravity
            shape['vel'].y += self.GRAVITY
            
            # Apply magnetic force
            if self.player_magnet['strength'] != 0:
                vec_to_magnet = self.player_magnet['pos'] - shape['pos']
                dist_sq = vec_to_magnet.length_squared()
                if dist_sq > 1: # Avoid division by zero and extreme forces
                    force_mag = self.player_magnet['strength'] * self.MAGNET_FORCE_SCALE / max(dist_sq, 100)
                    force_vec = vec_to_magnet.normalize() * force_mag
                    shape['vel'] += force_vec
            
            # Apply damping
            shape['vel'] *= self.SHAPE_DAMPING
            
            # Update position
            shape['pos'] += shape['vel']

            # Continuous reward for approaching target
            target_slot = self.slots[shape['target_slot_idx']]
            current_dist = shape['pos'].distance_to(target_slot['pos'])
            reward += (shape['prev_dist_to_slot'] - current_dist) * 0.01 # Small reward for getting closer
            shape['prev_dist_to_slot'] = current_dist
            
            # Penalty for approaching edge
            dist_to_edge_x = min(shape['pos'].x, self.WIDTH - shape['pos'].x)
            if dist_to_edge_x < 50:
                reward -= (50 - dist_to_edge_x) * 0.002

            # Check for placement
            if not shape['is_placed'] and target_slot['occupied_by'] == -1:
                if shape['pos'].distance_to(target_slot['pos']) < (shape['radius'] + target_slot['radius']) / 2:
                    shape['is_placed'] = True
                    shape['pos'] = target_slot['pos'].copy() # Snap to slot
                    target_slot['occupied_by'] = i
                    self.score += 1
                    reward += 10
                    # Sound: Success chime
                    self._create_particles(target_slot['pos'], shape['color'], 30, 3.0)
            
            # Check for out of bounds
            if shape['pos'].y > self.HEIGHT + shape['radius']:
                self.game_over = True
                reward -= 10
                # Sound: Failure buzz

        # --- Update Particles ---
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = self.game_over or self.timer <= 0 or self.score == len(self.shapes)
        
        if terminated and self.score == len(self.shapes):
            reward += 100 # Victory bonus
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    # --- Rendering Methods ---
    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        self._render_particles()
        self._render_slots()
        self._render_shapes()
        self._render_magnet()

    def _render_slots(self):
        for slot in self.slots:
            self._draw_shape(self.screen, slot['type'], slot['pos'], slot['radius'], slot['color'], filled=False)

    def _render_shapes(self):
        for shape in self.shapes:
            self._draw_shape(self.screen, shape['type'], shape['pos'], shape['radius'], shape['color'], filled=True)
            # Add metallic sheen
            sheen_pos = shape['pos'] + pygame.Vector2(-4, -4)
            sheen_radius = shape['radius'] * 0.4
            pygame.draw.circle(self.screen, (255, 255, 255, 150), (int(sheen_pos.x), int(sheen_pos.y)), int(sheen_radius))

    def _render_magnet(self):
        strength = self.player_magnet['strength']
        if strength == 0:
            return

        pos = self.player_magnet['pos']
        radius = self.player_magnet['radius']
        color = self.COLOR_MAGNET_ATTRACT if strength > 0 else self.COLOR_MAGNET_REPEL
        
        # Pulsating field lines
        num_lines = 12
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        max_field_radius = 80 + 30 * pulse
        
        for i in range(num_lines):
            angle = (i / num_lines) * 2 * math.pi
            start_radius = radius + 5
            end_radius = max_field_radius
            
            start_point = pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * start_radius
            end_point = pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * end_radius
            
            alpha = 100 - (100 * (i/num_lines))
            line_color = (*color, int(alpha * pulse))
            
            pygame.draw.aaline(self.screen, line_color, start_point, end_point)

        # Draw magnet core glow
        for i in range(int(radius), 0, -2):
            alpha = 150 * (1 - i / radius)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), i, (*color, int(alpha)))
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        
    def _render_ui(self):
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"PLACED: {self.score} / {len(self.shapes)}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

    # --- Helper Methods ---
    def _draw_shape(self, surface, shape_type, pos, radius, color, filled=True):
        x, y = int(pos.x), int(pos.y)
        width = 3 if not filled else 0
        
        if shape_type == 'circle':
            pygame.draw.circle(surface, color, (x, y), int(radius), width)
        elif shape_type == 'square':
            pygame.draw.rect(surface, color, (x - radius, y - radius, 2 * radius, 2 * radius), width)
        elif shape_type == 'triangle':
            points = [
                (x, y - radius),
                (x - radius, y + radius * 0.7),
                (x + radius, y + radius * 0.7)
            ]
            pygame.draw.polygon(surface, color, points, width)
        elif shape_type == 'diamond':
            points = [
                (x, y - radius),
                (x - radius, y),
                (x, y + radius),
                (x + radius, y)
            ]
            pygame.draw.polygon(surface, color, points, width)
        elif shape_type == 'pentagon':
            points = []
            for i in range(5):
                angle = -math.pi / 2 + (2 * math.pi * i) / 5
                points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
            pygame.draw.polygon(surface, color, points, width)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 41),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['lifespan'] / 40)
            color = (*p['color'], alpha)
            size = max(1, int(3 * (p['lifespan'] / 40)))
            pygame.draw.circle(self.screen, color, p['pos'], size)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this on an instance of the class to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space from a reset state
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    
    # Run validation on the created instance
    # env.validate_implementation() # Commented out for headless execution
    
    obs, info = env.reset()
    
    # Set up display only if not running in a truly headless environment
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Magnetic Shape Sorter")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Use arrow keys to move the magnet.")
        print("Hold SPACE to attract shapes.")
        print("Hold SHIFT to repel shapes.")
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Draw the observation from the environment to the display screen
            # Need to transpose it back to Pygame's (width, height, channels) format
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                pygame.time.wait(2000) # Pause before restarting

            clock.tick(GameEnv.FPS)
        
    env.close()