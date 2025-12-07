import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Gravity Orbs'.

    The player tilts the environment to guide 5 orbs to a goal area at the
    bottom of the screen. Orbs can collide with special squares to become
    larger, exerting a gravitational pull on other orbs, creating chain
    reactions. The goal is to save all orbs within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Tilt the world to guide all orbs to the goal at the bottom. "
        "Collide with special squares to make orbs larger, creating a gravitational pull on others."
    )
    user_guide = "Use the arrow keys (↑↓←→) to tilt the environment and guide the orbs to the goal."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_ORB_PALETTE = [
        (255, 87, 34),   # Deep Orange
        (3, 169, 244),   # Light Blue
        (255, 235, 59),  # Yellow
        (233, 30, 99),   # Pink
        (76, 175, 80)    # Green
    ]
    COLOR_SQUARE = (0, 255, 150)
    COLOR_LARGE_ORB = (171, 71, 188)
    COLOR_GRAVITY_WELL = (171, 71, 188)
    COLOR_GOAL = (0, 255, 150, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    TILT_FORCE = 0.15
    GRAVITY = 0.03
    DAMPING = 0.995
    MIN_VELOCITY = 0.01
    LARGE_ORB_GRAVITY_PULL = 2.0
    LARGE_ORB_GRAVITY_RADIUS = 150

    # Game Objects
    NUM_ORBS = 5
    NUM_SQUARES = 3
    ORB_BASE_RADIUS = 10
    SQUARE_SIZE = 20
    GOAL_Y = SCREEN_HEIGHT - 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = self.SCREEN_WIDTH
        self.screen_height = self.SCREEN_HEIGHT

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        self._generate_background()

        # --- Game State Initialization ---
        self.orbs = []
        self.squares = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.previous_orb_y = []
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use a numpy random generator for consistency
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.particles = []

        # --- Create Game Objects ---
        self.orbs = [self._create_orb(i) for i in range(self.NUM_ORBS)]
        self.squares = self._create_squares()

        self.previous_orb_y = [orb['pos'][1] for orb in self.orbs]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1 # Not used
        # shift_held = action[2] == 1 # Not used

        # --- Game Logic ---
        self.steps += 1
        self.time_left -= 1
        
        reward = self._update_physics_and_state(movement)
        self.score += reward
        
        # --- Termination ---
        terminated = self._check_termination()
        truncated = False # No truncation condition in this game
        if terminated:
            self.game_over = True
            if self.time_left <= 0:
                reward -= 100 # Timeout penalty
            else: # Won
                reward += 100 # Win bonus
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    # --- Private Helper Methods ---

    def _update_physics_and_state(self, movement):
        step_reward = 0.0

        # 1. Calculate forces
        forces = self._calculate_forces(movement)

        # 2. Update orbs
        for i, orb in enumerate(self.orbs):
            if orb['is_at_goal']:
                continue

            # Apply force
            accel = forces[i] / orb['mass']
            orb['vel'] += accel
            
            # Apply damping
            orb['vel'] *= self.DAMPING
            
            # Update position
            orb['pos'] += orb['vel']

            # Reward for moving down
            if orb['pos'][1] > self.previous_orb_y[i]:
                step_reward += 0.1
            self.previous_orb_y[i] = orb['pos'][1]

        # 3. Handle collisions
        self._handle_wall_collisions()
        self._handle_orb_collisions()
        step_reward += self._handle_square_collisions()
        
        # 4. Check goal
        step_reward += self._check_goal_reached()

        # 5. Update particles
        self._update_particles()
        
        return step_reward

    def _calculate_forces(self, movement):
        forces = [np.array([0.0, 0.0]) for _ in self.orbs]
        
        # Tilt force
        tilt_force_vec = np.array([0.0, 0.0])
        if movement == 1: # Up
            tilt_force_vec[1] = -self.TILT_FORCE
        elif movement == 2: # Down
            tilt_force_vec[1] = self.TILT_FORCE
        elif movement == 3: # Left
            tilt_force_vec[0] = -self.TILT_FORCE
        elif movement == 4: # Right
            tilt_force_vec[0] = self.TILT_FORCE

        # Apply forces to all orbs
        for i, orb in enumerate(self.orbs):
            if orb['is_at_goal']:
                continue
            # Constant gravity
            forces[i][1] += self.GRAVITY * orb['mass']
            # Player tilt
            forces[i] += tilt_force_vec * orb['mass']

        # Large orb gravitational pull
        large_orbs = [o for o in self.orbs if o['is_large'] and not o['is_at_goal']]
        small_orbs_indices = [i for i, o in enumerate(self.orbs) if not o['is_large'] and not o['is_at_goal']]

        for large_orb in large_orbs:
            for i in small_orbs_indices:
                small_orb = self.orbs[i]
                dist_vec = large_orb['pos'] - small_orb['pos']
                dist_sq = np.dot(dist_vec, dist_vec)

                if 0 < dist_sq < self.LARGE_ORB_GRAVITY_RADIUS**2:
                    dist = math.sqrt(dist_sq)
                    direction = dist_vec / dist
                    # Gravity falls off with square of distance
                    strength = self.LARGE_ORB_GRAVITY_PULL * large_orb['mass'] / dist_sq
                    force_on_small = direction * strength
                    forces[i] += force_on_small
        
        return forces

    def _handle_wall_collisions(self):
        for orb in self.orbs:
            if orb['is_at_goal']:
                continue
            
            # Left/Right walls
            if orb['pos'][0] - orb['radius'] < 0:
                orb['pos'][0] = orb['radius']
                orb['vel'][0] *= -0.9 # Lose some energy on bounce
            elif orb['pos'][0] + orb['radius'] > self.screen_width:
                orb['pos'][0] = self.screen_width - orb['radius']
                orb['vel'][0] *= -0.9
            
            # Top wall
            if orb['pos'][1] - orb['radius'] < 0:
                orb['pos'][1] = orb['radius']
                orb['vel'][1] *= -0.9

    def _handle_orb_collisions(self):
        for i in range(len(self.orbs)):
            for j in range(i + 1, len(self.orbs)):
                orb1 = self.orbs[i]
                orb2 = self.orbs[j]

                if orb1['is_at_goal'] or orb2['is_at_goal']:
                    continue

                dist_vec = orb1['pos'] - orb2['pos']
                dist_sq = np.dot(dist_vec, dist_vec)
                min_dist = orb1['radius'] + orb2['radius']

                if dist_sq < min_dist**2 and dist_sq > 0:
                    # Collision detected
                    dist = math.sqrt(dist_sq)
                    
                    # Resolve overlap
                    overlap = min_dist - dist
                    normal = dist_vec / dist
                    orb1['pos'] += normal * overlap / 2
                    orb2['pos'] -= normal * overlap / 2
                    
                    # Elastic collision response
                    v1 = orb1['vel']
                    v2 = orb2['vel']
                    m1 = orb1['mass']
                    m2 = orb2['mass']
                    x1 = orb1['pos']
                    x2 = orb2['pos']

                    # Simplified elastic collision formula
                    new_v1 = v1 - (2*m2/(m1+m2)) * np.dot(v1-v2, x1-x2) / np.dot(x1-x2, x1-x2) * (x1-x2)
                    new_v2 = v2 - (2*m1/(m1+m2)) * np.dot(v2-v1, x2-x1) / np.dot(x2-x1, x2-x1) * (x2-x1)

                    orb1['vel'] = new_v1
                    orb2['vel'] = new_v2

                    self._create_collision_particles( (orb1['pos'] + orb2['pos']) / 2, 5)

    def _handle_square_collisions(self):
        reward = 0
        for orb in self.orbs:
            if orb['is_at_goal'] or orb['is_large']:
                continue
            
            orb_rect = pygame.Rect(orb['pos'][0] - orb['radius'], orb['pos'][1] - orb['radius'], orb['radius']*2, orb['radius']*2)
            
            for i, square in reversed(list(enumerate(self.squares))):
                if orb_rect.colliderect(square):
                    if not orb['is_large']:
                        orb['is_large'] = True
                        orb['radius'] *= 1.5
                        orb['mass'] = orb['radius']**2
                        orb['color'] = self.COLOR_LARGE_ORB
                        self._create_collision_particles(orb['pos'], 20, self.COLOR_SQUARE)
                        reward += 1.0
                        self.squares.pop(i)
                        break
        return reward

    def _check_goal_reached(self):
        reward = 0
        for orb in self.orbs:
            if not orb['is_at_goal'] and orb['pos'][1] > self.GOAL_Y:
                orb['is_at_goal'] = True
                orb['vel'] = np.array([0.0, 0.0])
                reward += 5.0
                self._create_collision_particles(orb['pos'], 30, self.COLOR_GOAL)
        return reward

    def _check_termination(self):
        # Win condition: all orbs are at the goal
        if all(o['is_at_goal'] for o in self.orbs):
            return True
        # Loss condition: time runs out
        if self.time_left <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        orbs_saved = sum(1 for o in self.orbs if o['is_at_goal'])
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "orbs_saved": orbs_saved
        }

    def _render_game(self):
        # Render gravity wells for large orbs
        for orb in self.orbs:
            if orb['is_large'] and not orb['is_at_goal']:
                radius = self.LARGE_ORB_GRAVITY_RADIUS
                center = (int(orb['pos'][0]), int(orb['pos'][1]))
                for i in range(10):
                    alpha = 15 - i * 1.5
                    if alpha > 0:
                        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius * (i / 10)), (*self.COLOR_GRAVITY_WELL, int(alpha)))
        
        # Render squares
        for square in self.squares:
            pygame.draw.rect(self.screen, self.COLOR_SQUARE, square)

        # Render orbs
        for orb in self.orbs:
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            radius = int(orb['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, orb['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, orb['color'])

        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        # Goal line
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (0, self.GOAL_Y, self.screen_width, self.screen_height - self.GOAL_Y))

        # Timer
        time_str = f"Time: {max(0, self.time_left / self.FPS):.1f}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.screen_width - time_surf.get_width() - 15, 10))
        
        # Orbs saved
        orbs_saved = sum(1 for o in self.orbs if o['is_at_goal'])
        saved_str = f"Saved: {orbs_saved} / {self.NUM_ORBS}"
        saved_surf = self.font_large.render(saved_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(saved_surf, (15, self.screen_height - saved_surf.get_height() - 5))

    def _generate_background(self):
        self.background_surface = pygame.Surface((self.screen_width, self.screen_height))
        for y in range(self.screen_height):
            # Linear interpolation between top and bottom colors
            ratio = y / self.screen_height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.background_surface, color, (0, y), (self.screen_width, y))

    def _create_orb(self, index):
        return {
            'pos': np.array([
                self.np_random.uniform(self.ORB_BASE_RADIUS, self.screen_width - self.ORB_BASE_RADIUS),
                self.np_random.uniform(self.ORB_BASE_RADIUS, self.screen_height / 2)
            ], dtype=float),
            'vel': np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)], dtype=float),
            'color': self.COLOR_ORB_PALETTE[index % len(self.COLOR_ORB_PALETTE)],
            'radius': self.ORB_BASE_RADIUS,
            'mass': self.ORB_BASE_RADIUS**2,
            'is_large': False,
            'is_at_goal': False
        }

    def _create_squares(self):
        squares = []
        attempts = 0
        while len(squares) < self.NUM_SQUARES and attempts < 1000:
            attempts += 1
            s = pygame.Rect(
                self.np_random.uniform(self.SQUARE_SIZE, self.screen_width - self.SQUARE_SIZE),
                self.np_random.uniform(self.screen_height * 0.2, self.screen_height * 0.8 - self.SQUARE_SIZE),
                self.SQUARE_SIZE, self.SQUARE_SIZE
            )
            # Ensure no overlap with other squares
            if not any(s.colliderect(other) for other in squares):
                squares.append(s)
        return squares

    def _create_collision_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifetime': self.np_random.integers(15, 31),
                'radius': self.np_random.uniform(1, 3),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0.5]

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Orbs")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while True:
        # --- Human Input ---
        action = [0, 0, 0] # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated:
            # Display Game Over message
            font = pygame.font.SysFont("Arial", 50, bold=True)
            orbs_saved = info.get('orbs_saved', 0)
            if orbs_saved == GameEnv.NUM_ORBS:
                text = "YOU WIN!"
                color = (100, 255, 100)
            else:
                text = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2))
            screen.blit(text_surf, text_rect)

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()