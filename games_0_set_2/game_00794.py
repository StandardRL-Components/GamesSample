
# Generated: 2025-08-27T14:47:55.231702
# Source Brief: brief_00794.md
# Brief Index: 794

        
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
        "Controls: Arrow keys to move the crystal on the grid. Match the crystal to a slot to fill it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Navigate the crystal to fill all colored slots before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.TOTAL_SLOTS = 10
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLORS = {
            'BG': (25, 30, 45),
            'GRID': (40, 45, 60),
            'SLOT_OUTLINE': (200, 200, 220),
            'CRYSTAL_OUTLINE': (255, 255, 255),
            'TEXT': (220, 220, 240),
            'RED': (255, 80, 80),
            'GREEN': (80, 255, 80),
            'BLUE': (80, 120, 255),
            'YELLOW': (255, 255, 80),
            'PURPLE': (200, 80, 255),
        }
        self.PALETTE = [self.COLORS['RED'], self.COLORS['GREEN'], self.COLORS['BLUE'], self.COLORS['YELLOW'], self.COLORS['PURPLE']]
        self.PALETTE_NAMES = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'PURPLE']
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.crystal_pos = None
        self.crystal_color_idx = None
        self.slots = []
        self.particles = []
        self.last_reward = 0
        self.grid_offset = (self.WIDTH // 2, self.HEIGHT // 2 - self.GRID_SIZE * self.TILE_HEIGHT // 4)

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _setup_level(self):
        # Generate 10 unique slot positions
        occupied_positions = set()
        self.slots = []
        while len(self.slots) < self.TOTAL_SLOTS:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if pos not in occupied_positions:
                occupied_positions.add(pos)
                color_idx = self.np_random.integers(0, len(self.PALETTE))
                self.slots.append({
                    "pos": pos,
                    "color_idx": color_idx,
                    "filled": False
                })

        # Place the crystal ensuring a valid starting move
        while True:
            # Pick a random slot to determine the crystal's initial color
            initial_slot = self.np_random.choice(self.slots)
            self.crystal_color_idx = initial_slot['color_idx']

            # Generate a random starting position for the crystal
            start_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            
            # Ensure start pos is not on an existing slot
            if start_pos in occupied_positions:
                continue

            # Find the nearest slot of the same color
            min_dist = float('inf')
            for slot in self.slots:
                if not slot['filled'] and slot['color_idx'] == self.crystal_color_idx:
                    dist = self._get_manhattan_distance(start_pos, slot['pos'])
                    min_dist = min(min_dist, dist)
            
            # If a fair start is found (match within 3 moves), accept and break
            if min_dist <= 3:
                self.crystal_pos = start_pos
                break
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.particles = []
        self.last_reward = 0
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        reward = 0
        
        prev_pos = self.crystal_pos
        dx, dy = 0, 0
        moved = False

        if movement == 1: # Up
            dy = -1
        elif movement == 2: # Down
            dy = 1
        elif movement == 3: # Left
            dx = -1
        elif movement == 4: # Right
            dx = 1
        
        if dx != 0 or dy != 0:
            new_pos = (self.crystal_pos[0] + dx, self.crystal_pos[1] + dy)
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.crystal_pos = new_pos
                self.moves_remaining -= 1
                moved = True
                # SFX: Crystal move sound

        # Continuous reward for moving towards a goal
        if moved:
            # Find closest unfilled slot of current crystal color
            dist_before = float('inf')
            dist_after = float('inf')
            
            matching_slots = [s for s in self.slots if not s['filled'] and s['color_idx'] == self.crystal_color_idx]
            if matching_slots:
                for slot in matching_slots:
                    dist_before = min(dist_before, self._get_manhattan_distance(prev_pos, slot['pos']))
                    dist_after = min(dist_after, self._get_manhattan_distance(self.crystal_pos, slot['pos']))

                if dist_after < dist_before:
                    reward += 1.0  # Moved closer
                elif dist_after > dist_before:
                    reward -= 0.1 # Moved further
        
        # Check for slot interaction
        for slot in self.slots:
            if self.crystal_pos == slot['pos'] and not slot['filled']:
                self.crystal_color_idx = slot['color_idx']
                slot['filled'] = True
                self.score += 10
                reward += 10.0
                self._create_particles(self.crystal_pos, self.PALETTE[self.crystal_color_idx])
                # SFX: Match success sound
                break

        self.last_reward = reward
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            num_filled = sum(1 for s in self.slots if s['filled'])
            if num_filled == self.TOTAL_SLOTS:
                self.score += 100
                self.last_reward += 100 # Goal-oriented reward
                # SFX: Level complete fanfare
            else:
                self.score -= 50
                self.last_reward -= 50 # Penalty for failure
                # SFX: Game over sound
        
        return self._get_observation(), self.last_reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.moves_remaining <= 0:
            return True
        if all(slot['filled'] for slot in self.slots):
            return True
        return False

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.grid_offset[0] + (grid_x - grid_y) * self.TILE_WIDTH / 2
        screen_y = self.grid_offset[1] + (grid_x + grid_y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _render_iso_poly(self, grid_x, grid_y, color, filled=True, outline_color=None, outline_width=2):
        center_x, center_y = self._iso_to_screen(grid_x, grid_y)
        points = [
            (center_x, center_y - self.TILE_HEIGHT // 2),
            (center_x + self.TILE_WIDTH // 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT // 2),
            (center_x - self.TILE_WIDTH // 2, center_y),
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _create_particles(self, grid_pos, color):
        screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            radius = random.uniform(2, 5)
            self.particles.append([list(screen_pos), velocity, radius, lifetime, color])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0][0] += p[1][0] # Update pos x
            p[0][1] += p[1][1] # Update pos y
            p[3] -= 1 # Decrement lifetime
            if p[3] > 0:
                alpha = int(255 * (p[3] / 40))
                color = (*p[4], alpha)
                temp_surf = pygame.Surface((p[2]*2, p[2]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p[2], p[2]), p[2])
                self.screen.blit(temp_surf, (p[0][0] - p[2], p[0][1] - p[2]))
                active_particles.append(p)
        self.particles = active_particles

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._render_iso_poly(c, r, self.COLORS['GRID'], outline_color=(50, 55, 70))
        
        # Draw slots
        for slot in self.slots:
            pos = slot['pos']
            color = self.PALETTE[slot['color_idx']]
            if slot['filled']:
                # Glowing effect
                glow_surf = pygame.Surface((self.TILE_WIDTH*1.5, self.TILE_HEIGHT*1.5), pygame.SRCALPHA)
                glow_color = (*color, 70)
                pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
                screen_pos = self._iso_to_screen(pos[0], pos[1])
                self.screen.blit(glow_surf, (screen_pos[0] - self.TILE_WIDTH*0.75, screen_pos[1] - self.TILE_HEIGHT*0.75))
                self._render_iso_poly(pos[0], pos[1], color)
            else:
                self._render_iso_poly(pos[0], pos[1], self.COLORS['GRID'], outline_color=self.COLORS['SLOT_OUTLINE'])

        # Draw particles
        self._update_and_draw_particles()

        # Draw crystal
        crystal_screen_pos = self._iso_to_screen(self.crystal_pos[0], self.crystal_pos[1])
        crystal_color = self.PALETTE[self.crystal_color_idx]
        radius = self.TILE_HEIGHT // 2
        
        # Crystal glow
        glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*crystal_color, 100), (radius*2, radius*2), radius*2)
        pygame.draw.circle(glow_surf, (*crystal_color, 50), (radius*2, radius*2), radius*1.5)
        self.screen.blit(glow_surf, (crystal_screen_pos[0] - radius*2, crystal_screen_pos[1] - radius*2))
        
        # Crystal body
        pygame.gfxdraw.filled_circle(self.screen, crystal_screen_pos[0], crystal_screen_pos[1], radius - 2, crystal_color)
        pygame.gfxdraw.aacircle(self.screen, crystal_screen_pos[0], crystal_screen_pos[1], radius - 2, self.COLORS['CRYSTAL_OUTLINE'])

    def _render_text(self, text, font, color, position, anchor="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, anchor, position)
        self.screen.blit(text_surface, text_rect)

    def _render_ui(self):
        # Moves remaining
        self._render_text(f"Moves: {self.moves_remaining}", self.font_small, self.COLORS['TEXT'], (10, 10))
        
        # Score
        self._render_text(f"Score: {self.score}", self.font_small, self.COLORS['TEXT'], (self.WIDTH - 10, 10), anchor="topright")
        
        # Filled slots
        filled_count = sum(1 for s in self.slots if s['filled'])
        self._render_text(f"Slots: {filled_count}/{self.TOTAL_SLOTS}", self.font_small, self.COLORS['TEXT'], (10, 35))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            num_filled = sum(1 for s in self.slots if s['filled'])
            if num_filled == self.TOTAL_SLOTS:
                msg = "LEVEL COMPLETE!"
            else:
                msg = "GAME OVER"
            self._render_text(msg, self.font_large, self.COLORS['YELLOW'], (self.WIDTH // 2, self.HEIGHT // 2), anchor="center")

    def _get_observation(self):
        self.screen.fill(self.COLORS['BG'])
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "filled_slots": sum(1 for s in self.slots if s['filled']),
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op, release space, release shift
    
    while not done:
        # Get observation from the environment
        obs_pygame = np.transpose(env._get_observation(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_pygame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action = np.array([0, 0, 0])
                    continue

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

                # Reset action after processing
                action = np.array([0, 0, 0])

        clock.tick(30) # Limit frame rate
        
    env.close()
    pygame.quit()