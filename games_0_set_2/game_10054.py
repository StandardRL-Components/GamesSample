import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:50:02.743393
# Source Brief: brief_00054.md
# Brief Index: 54
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Evo-Cell Survival Environment for Gymnasium.

    The player controls a cell that must absorb smaller cells to grow and evolve.
    It must evade predatory cells and navigate around static obstacles.
    The goal is to reach Level 5.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Used for splitting ability
    - actions[2]: Shift button (0=released, 1=held) -> Used for rejoining ability

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.

    Rewards:
    - +0.1 for absorbing a food cell
    - -0.01 per step (encourages efficiency)
    - +1.0 for leveling up
    - +100 for winning the game (reaching Level 5)
    - -100 for being consumed by a predator
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a cell that grows by absorbing food. Evade larger predators and navigate obstacles to evolve and win."
    )
    user_guide = (
        "Use arrow keys to move. At Level 4, press Space to split into two cells. Press Shift to rejoin."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Pygame Setup ===
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # === Colors ===
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_FOOD = (150, 255, 150)
        self.COLOR_PREDATOR = (255, 80, 80)
        self.COLOR_PREDATOR_GLOW = (255, 80, 80, 70)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # === Game Parameters ===
        self.MAX_STEPS = 1500
        self.LEVEL_THRESHOLDS = {1: 15, 2: 25, 3: 40, 4: 60, 5: float('inf')}
        self.BASE_PLAYER_SPEED = 3.0
        self.BASE_PREDATOR_SPEED = 1.5

        # === State Variables ===
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self.player_pos = None
        self.player_size = None
        self.player_speed = None
        self.player_split = False
        self.sub_cells = []

        self.food_items = []
        self.predators = []
        self.obstacles = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        # Initialize state
        # self.reset() # reset is called by the wrapper
        
        # === Final Validation ===
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_size = 10
        self.player_split = False
        self.sub_cells = []

        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Step penalty
        self.steps += 1

        self._handle_player_actions(action)
        self._update_predators()
        
        collisions = self._check_collisions()
        reward += collisions['reward']
        
        if not self.game_over: # Can't level up if already dead
            level_up = self._check_level_up()
            if level_up:
                reward += 1.0
                if self.level == 5:
                    self.game_over = True
                    reward += 100.0 # Victory reward
                else:
                    self._setup_level() # Reset for new level

        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        truncated = self.steps >= self.MAX_STEPS
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_level(self):
        self.food_items.clear()
        self.predators.clear()
        self.obstacles.clear()

        # Player stats
        self.player_speed = self.BASE_PLAYER_SPEED * (1 + 0.5 * (self.level > 1))

        # Obstacles
        num_obstacles = 4 + (self.level - 1) * 2
        for _ in range(num_obstacles):
            w, h = self.np_random.integers(20, 80, size=2)
            x = self.np_random.integers(50, self.WIDTH - w - 50)
            y = self.np_random.integers(50, self.HEIGHT - h - 50)
            obstacle_rect = pygame.Rect(x, y, w, h)
            # Ensure no obstacle on player start
            if not obstacle_rect.collidepoint(tuple(self.player_pos)):
                self.obstacles.append(obstacle_rect)

        # Food
        num_food = 25 - (self.level - 1) * 5
        for _ in range(num_food):
            self._spawn_entity(self.food_items, self.np_random.integers(3, 5))

        # Predators
        num_predators = 2 + self.level
        predator_speed = self.BASE_PREDATOR_SPEED + (self.level - 1) * 0.2
        for _ in range(num_predators):
            predator = self._spawn_entity({}, 15)
            angle = self.np_random.uniform(0, 2 * math.pi)
            predator['vel'] = np.array([math.cos(angle), math.sin(angle)]) * predator_speed
            self.predators.append(predator)

    def _spawn_entity(self, collection, size):
        while True:
            pos = np.array([
                self.np_random.uniform(size, self.WIDTH - size),
                self.np_random.uniform(size, self.HEIGHT - size)
            ], dtype=float)
            
            is_colliding = False
            for obs in self.obstacles:
                if obs.collidepoint(tuple(pos)):
                    is_colliding = True
                    break
            if not is_colliding:
                if isinstance(collection, list):
                    collection.append({'pos': pos, 'size': size})
                else: # is a dict for a single entity
                    collection['pos'] = pos
                    collection['size'] = size
                return collection

    def _handle_player_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Abilities (Split/Join) ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed and self.level >= 4 and not self.player_split:
            # SFX: cell_split.wav
            self.player_split = True
            self.player_size /= 2
            angle = self.np_random.uniform(0, 2 * math.pi)
            offset = np.array([math.cos(angle), math.sin(angle)]) * (self.player_size + 2)
            self.sub_cells = [
                {'pos': self.player_pos + offset, 'size': self.player_size},
                {'pos': self.player_pos - offset, 'size': self.player_size}
            ]
        elif shift_pressed and self.player_split:
            dist = np.linalg.norm(self.sub_cells[0]['pos'] - self.sub_cells[1]['pos'])
            if dist < (self.sub_cells[0]['size'] + self.sub_cells[1]['size']) * 1.5:
                # SFX: cell_rejoin.wav
                self.player_pos = (self.sub_cells[0]['pos'] + self.sub_cells[1]['pos']) / 2
                self.player_size *= 2
                self.player_split = False
                self.sub_cells = []

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Movement ---
        direction = np.array([0, 0], dtype=float)
        if movement == 1: direction[1] = -1  # Up
        elif movement == 2: direction[1] = 1   # Down
        elif movement == 3: direction[0] = -1  # Left
        elif movement == 4: direction[0] = 1   # Right

        if np.any(direction):
            if not self.player_split:
                self._move_entity(self.player_pos, direction * self.player_speed, self.player_size)
            else:
                for cell in self.sub_cells:
                    self._move_entity(cell['pos'], direction * self.player_speed, cell['size'])
    
    def _move_entity(self, pos, vel, size):
        new_pos = pos + vel

        # Boundary checks
        new_pos[0] = np.clip(new_pos[0], size, self.WIDTH - size)
        new_pos[1] = np.clip(new_pos[1], size, self.HEIGHT - size)

        # Obstacle collision
        temp_rect = pygame.Rect(0, 0, int(size * 2), int(size * 2))
        temp_rect.center = (int(new_pos[0]), int(new_pos[1]))
        
        collided = False
        for obs in self.obstacles:
            if obs.colliderect(temp_rect):
                collided = True
                break
        
        if not collided:
            pos[:] = new_pos
        return collided

    def _update_predators(self):
        for predator in self.predators:
            if self._move_entity(predator['pos'], predator['vel'], predator['size']):
                # Hit obstacle/wall, reverse direction
                predator['vel'] *= -1
            # Add slight random turning to prevent getting stuck
            if self.np_random.random() < 0.02:
                angle = self.np_random.uniform(-0.3, 0.3)
                rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                predator['vel'] = np.dot(rot_matrix, predator['vel'])

    def _check_collisions(self):
        reward = 0
        
        player_entities = self.sub_cells if self.player_split else [{'pos': self.player_pos, 'size': self.player_size}]

        # Player vs Food
        eaten_food_indices = []
        for i, food in enumerate(self.food_items):
            for p_cell in player_entities:
                dist = np.linalg.norm(p_cell['pos'] - food['pos'])
                if dist < p_cell['size'] + food['size']:
                    if i not in eaten_food_indices:
                        eaten_food_indices.append(i)
                        
        if eaten_food_indices:
            # SFX: absorb.wav
            for i in sorted(eaten_food_indices, reverse=True):
                self._create_particles(self.food_items[i]['pos'], self.COLOR_FOOD)
                del self.food_items[i]
                
            growth_amount = len(eaten_food_indices) * 1.0 # Grow by 10% of base size
            self.player_size += growth_amount
            if self.player_split:
                for cell in self.sub_cells:
                    cell['size'] += growth_amount / 2
            
            reward += 0.1 * len(eaten_food_indices)

        # Player vs Predator
        for predator in self.predators:
            for p_cell in player_entities:
                dist = np.linalg.norm(p_cell['pos'] - predator['pos'])
                if dist < p_cell['size'] + predator['size']:
                    # SFX: player_death.wav
                    self.game_over = True
                    reward -= 100.0
                    return {'reward': reward}

        return {'reward': reward}

    def _check_level_up(self):
        if self.player_size >= self.LEVEL_THRESHOLDS.get(self.level, float('inf')):
            # SFX: level_up.wav
            self.level += 1
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            pygame.draw.rect(self.screen, tuple(int(c*0.8) for c in self.COLOR_OBSTACLE), obs, 2)

        # Draw food
        for food in self.food_items:
            pos = food['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(food['size']), self.COLOR_FOOD)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(food['size']), self.COLOR_FOOD)

        # Draw predators
        for predator in self.predators:
            self._draw_glow_circle(predator['pos'], predator['size'] * 1.5, self.COLOR_PREDATOR_GLOW)
            self._draw_oriented_triangle(predator['pos'], predator['size'], predator['vel'], self.COLOR_PREDATOR)
            
        # Draw player
        if not self.player_split:
            self._draw_glow_circle(self.player_pos, self.player_size * 1.5, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(self.player_size), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(self.player_size), self.COLOR_PLAYER)
        else:
            for cell in self.sub_cells:
                self._draw_glow_circle(cell['pos'], cell['size'] * 1.5, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.aacircle(self.screen, int(cell['pos'][0]), int(cell['pos'][1]), int(cell['size']), self.COLOR_PLAYER)
                pygame.gfxdraw.filled_circle(self.screen, int(cell['pos'][0]), int(cell['pos'][1]), int(cell['size']), self.COLOR_PLAYER)

    def _render_ui(self):
        ui_panel = pygame.Surface((180, 85), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        mass_text_val = self.LEVEL_THRESHOLDS.get(self.level, 'WIN')
        mass_text = self.font_small.render(f"Mass: {self.player_size:.1f}/{mass_text_val}", True, self.COLOR_UI_TEXT)
        
        ui_panel.blit(level_text, (10, 5))
        ui_panel.blit(score_text, (10, 35))
        ui_panel.blit(mass_text, (10, 55))
        self.screen.blit(ui_panel, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_size": self.player_size,
        }

    def _draw_glow_circle(self, pos, radius, color):
        radius_int = int(radius)
        if radius_int <= 0: return
        surf = pygame.Surface((radius_int * 2, radius_int * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius_int, radius_int), radius_int)
        blit_pos = (int(pos[0] - radius_int), int(pos[1] - radius_int))
        self.screen.blit(surf, blit_pos, special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_oriented_triangle(self, pos, size, vel, color):
        angle = math.atan2(vel[1], vel[0])
        points = []
        for i in range(3):
            point_angle = angle + (i * 2 * math.pi / 3)
            if i == 0: # Point the nose
                p_size = size * 1.5
            else:
                p_size = size * 0.8
            x = pos[0] + p_size * math.cos(point_angle)
            y = pos[1] + p_size * math.sin(point_angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, int(255 * (p['life'] / 20)))
                color = (*p['color'], alpha)
                p_size_int = int(p['size'])
                if p_size_int <= 0: continue
                temp_surf = pygame.Surface((p_size_int*2, p_size_int*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p_size_int, p_size_int), p_size_int)
                blit_pos = tuple((p['pos'] - p['size']).astype(int))
                self.screen.blit(temp_surf, blit_pos, special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for manual play
    # We need to re-init display for manual play
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Evo-Cell Survival")
    clock = pygame.time.Clock()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if terminated:
            # Display game over message
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            text = font.render("GAME OVER (R to Restart)", True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            continue

        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != -0.01: # Print non-trivial rewards
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Clock ---
        clock.tick(30) # Run at 30 FPS

    env.close()