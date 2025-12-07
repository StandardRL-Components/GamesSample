
# Generated: 2025-08-27T19:37:06.047713
# Source Brief: brief_02206.md
# Brief Index: 2206

        
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
    """
    A Gymnasium environment for a top-down survival horror game.
    The player must navigate a spooky graveyard, collect 7 items,
    and avoid patrolling ghosts.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. "
        "Space and Shift have no effect in this version."
    )

    # Short, user-facing description of the game
    game_description = (
        "Survive the night in a haunted graveyard. Collect all 7 glowing items "
        "while evading the spectral ghosts that patrol the grounds."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, Pygame, and state variables.
        """
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Visuals & Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_FENCE = (40, 50, 70)
        self.COLOR_TOMBSTONE = (60, 70, 90)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_GHOST = (173, 216, 230)
        self.COLOR_GHOST_GLOW = (100, 150, 200)
        self.COLOR_ITEM = (50, 255, 50)
        self.COLOR_ITEM_GLOW = (150, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)
        
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_msg = pygame.font.Font(None, 72)
        except pygame.error:
            # Fallback if default font is not found
            self.font_ui = pygame.font.SysFont("sans", 28)
            self.font_msg = pygame.font.SysFont("sans", 72)


        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.PLAYER_SPEED = 3.5
        self.PLAYER_RADIUS = 8
        self.GHOST_RADIUS = 12
        self.GHOST_SPEED_NORMAL = 1.2
        self.GHOST_SPEED_SLOW = 0.6
        self.ITEM_RADIUS = 6
        self.NUM_ITEMS = 7
        self.NUM_GHOSTS = 3
        self.PROXIMITY_BONUS_RADIUS = 60

        # --- State Variables ---
        # These are initialized in reset()
        self.player_pos = None
        self.items = None
        self.ghosts = None
        self.obstacles = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rng = None

        # Call reset to set the initial state
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        
        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        self._generate_layout()
        self.player_pos = np.array([self.screen_width / 2, self.screen_height - 40.0])

        self._spawn_items()
        self._spawn_ghosts()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game by one timestep.
        
        Args:
            action (np.ndarray): The action from the MultiDiscrete action space.
        
        Returns:
            tuple: A 5-tuple containing (observation, reward, terminated, truncated, info).
        """
        if self.auto_advance:
            self.clock.tick(30) # Maintain 30 FPS

        reward = 0
        if self.game_over:
            # If game is over, do nothing but return the final state
            terminated = True
            return (
                self._get_observation(), 0, terminated, False, self._get_info()
            )

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        reward += 0.01  # Small survival reward per step

        # Player Movement
        prev_player_pos = self.player_pos.copy()
        self._move_player(movement)
        
        # Ghost Movement
        self._move_ghosts()

        # Particle Update
        self._update_particles()
        
        # --- Reward Calculation & State Changes ---
        # Proximity to nearest item
        dist_before, dist_after, nearest_item_pos = self._calculate_item_proximity(prev_player_pos)
        if nearest_item_pos is not None and dist_after > dist_before + 1: # Penalize moving away
             reward -= 0.02
        
        # Proximity to ghosts
        for ghost in self.ghosts:
            dist_to_ghost = np.linalg.norm(self.player_pos - ghost['pos'])
            if dist_to_ghost < self.PROXIMITY_BONUS_RADIUS:
                reward += 0.05 # Risky behavior bonus

        # Collisions
        # Player vs. Items
        collected_item_this_step = False
        for item in self.items:
            if not item['collected']:
                dist = np.linalg.norm(self.player_pos - item['pos'])
                if dist < self.PLAYER_RADIUS + self.ITEM_RADIUS:
                    item['collected'] = True
                    self.score += 10
                    reward += 10
                    collected_item_this_step = True
                    self._create_particles(item['pos'], self.COLOR_ITEM, 20)
                    # sfx: item_collect.wav

        # Player vs. Ghosts
        for ghost in self.ghosts:
            dist = np.linalg.norm(self.player_pos - ghost['pos'])
            if dist < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                self.game_over = True
                self.score -= 50
                reward = -50  # Overwrite other rewards with large penalty
                self._create_particles(self.player_pos, self.COLOR_PLAYER_GLOW, 30)
                # sfx: player_death.wav
                break
        
        # --- Termination Check ---
        num_collected = sum(1 for item in self.items if item['collected'])
        if not self.game_over and num_collected == self.NUM_ITEMS:
            self.game_over = True
            self.win = True
            self.score += 50
            reward += 50
            # sfx: victory.wav

        if not self.game_over and self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.score -= 10 # Penalty for running out of time
            reward -= 10
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "items_collected": sum(1 for item in self.items if item['collected']),
            "items_remaining": sum(1 for item in self.items if not item['collected']),
        }

    # --- Helper methods for game logic ---

    def _move_player(self, movement):
        """Moves the player and handles collisions."""
        original_pos = self.player_pos.copy()
        
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.screen_width - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.screen_height - self.PLAYER_RADIUS)

        # Obstacle collision
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        for obs in self.obstacles:
            if obs.colliderect(player_rect):
                self.player_pos = original_pos  # Revert move
                break

    def _move_ghosts(self):
        """Moves ghosts along their predefined paths."""
        speed = self.GHOST_SPEED_SLOW if self.steps < 30 else self.GHOST_SPEED_NORMAL
        for ghost in self.ghosts:
            target_pos = ghost['path'][ghost['waypoint_idx']]
            direction = target_pos - ghost['pos']
            dist = np.linalg.norm(direction)

            if dist < speed:
                ghost['pos'] = target_pos.copy()
                ghost['waypoint_idx'] = (ghost['waypoint_idx'] + 1) % len(ghost['path'])
            else:
                ghost['pos'] += (direction / dist) * speed
    
    def _calculate_item_proximity(self, prev_pos):
        """Calculates player's distance to the nearest uncollected item."""
        uncollected_items = [item['pos'] for item in self.items if not item['collected']]
        if not uncollected_items:
            return None, None, None

        distances = [np.linalg.norm(self.player_pos - pos) for pos in uncollected_items]
        distances_prev = [np.linalg.norm(prev_pos - pos) for pos in uncollected_items]
        
        min_idx = np.argmin(distances)
        
        return distances_prev[min_idx], distances[min_idx], uncollected_items[min_idx]

    # --- Helper methods for setup ---

    def _generate_layout(self):
        """Generates static obstacles (tombstones)."""
        self.obstacles = []
        # Create a border
        self.obstacles.append(pygame.Rect(0, 0, self.screen_width, 10))
        self.obstacles.append(pygame.Rect(0, self.screen_height - 10, self.screen_width, 10))
        self.obstacles.append(pygame.Rect(0, 0, 10, self.screen_height))
        self.obstacles.append(pygame.Rect(self.screen_width - 10, 0, 10, self.screen_height))
        
        # Generate some random tombstones
        for _ in range(10):
            w = self.rng.integers(20, 60)
            h = self.rng.integers(40, 80)
            x = self.rng.integers(40, self.screen_width - w - 40)
            y = self.rng.integers(40, self.screen_height - h - 40)
            self.obstacles.append(pygame.Rect(x, y, w, h))

    def _spawn_items(self):
        """Spawns items in valid locations."""
        self.items = []
        while len(self.items) < self.NUM_ITEMS:
            pos = np.array([
                self.rng.uniform(30, self.screen_width - 30),
                self.rng.uniform(30, self.screen_height - 30)
            ])
            item_rect = pygame.Rect(pos[0] - self.ITEM_RADIUS, pos[1] - self.ITEM_RADIUS, self.ITEM_RADIUS*2, self.ITEM_RADIUS*2)
            
            # Check for collision with obstacles or other items
            valid = True
            for obs in self.obstacles:
                if obs.colliderect(item_rect):
                    valid = False
                    break
            if not valid: continue
            
            for item in self.items:
                if np.linalg.norm(pos - item['pos']) < 50:
                    valid = False
                    break
            if not valid: continue
            
            self.items.append({'pos': pos, 'collected': False})

    def _spawn_ghosts(self):
        """Creates ghosts and their patrol paths."""
        self.ghosts = []
        paths = [
            # Path 1: Horizontal patrol top
            [np.array([100, 80.]), np.array([540, 80.])],
            # Path 2: Vertical patrol left
            [np.array([80, 120.]), np.array([80, 300.])],
            # Path 3: Box patrol right
            [np.array([450, 150.]), np.array([550, 150.]), np.array([550, 280.]), np.array([450, 280.])]
        ]
        for i in range(self.NUM_GHOSTS):
            path = paths[i % len(paths)]
            self.ghosts.append({
                'pos': path[0].copy(),
                'path': path,
                'waypoint_idx': 1
            })

    # --- Helper methods for rendering and effects ---

    def _render_game(self):
        """Renders all game elements."""
        # Draw fence/border
        pygame.draw.rect(self.screen, self.COLOR_FENCE, (0, 0, self.screen_width, self.screen_height), 10)
        
        # Draw tombstones
        for obs in self.obstacles[4:]: # Skip border rects
            pygame.draw.rect(self.screen, self.COLOR_TOMBSTONE, obs)
            pygame.draw.rect(self.screen, self.COLOR_FENCE, obs, 2) # Outline

        # Draw items
        for item in self.items:
            if not item['collected']:
                pos = (int(item['pos'][0]), int(item['pos'][1]))
                glow_radius = int(self.ITEM_RADIUS * 2.5 + 2 * math.sin(self.steps * 0.1))
                # Use a surface for transparency
                glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_ITEM_GLOW, 80), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (pos[0]-glow_radius, pos[1]-glow_radius))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, self.COLOR_ITEM)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, self.COLOR_ITEM)

        # Draw ghosts
        for ghost in self.ghosts:
            bob = math.sin(self.steps * 0.08 + id(ghost)) * 4
            pos = (int(ghost['pos'][0]), int(ghost['pos'][1] + bob))
            glow_radius = self.GHOST_RADIUS + 5
            # Surface for transparency
            ghost_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(ghost_surf, (*self.COLOR_GHOST, 100), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(ghost_surf, (*self.COLOR_GHOST, 150), (glow_radius, glow_radius), self.GHOST_RADIUS)
            self.screen.blit(ghost_surf, (pos[0]-glow_radius, pos[1]-glow_radius))

        # Draw particles
        for p in self.particles:
            p_color = (*p['color'], int(255 * p['life']))
            pygame.draw.circle(self.screen, p_color, p['pos'].astype(int), int(p['size']))

        # Draw player
        player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        glow_radius = int(self.PLAYER_RADIUS * 2.0)
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 120), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_pos_int[0]-glow_radius, player_pos_int[1]-glow_radius))
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        """Renders UI elements like score and messages."""
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Items remaining
        items_rem = sum(1 for item in self.items if not item['collected'])
        items_text = self.font_ui.render(f"Items: {items_rem}/{self.NUM_ITEMS}", True, self.COLOR_TEXT)
        self.screen.blit(items_text, (self.screen_width - items_text.get_width() - 15, 15))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg_text = self.font_msg.render("YOU SURVIVED!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_msg.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            text_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_text, text_rect)

    def _create_particles(self, pos, color, count):
        """Creates a burst of particles."""
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
                'life': self.rng.uniform(0.5, 1.0), # in seconds
                'size': self.rng.uniform(1, 4),
                'color': color
            })
    
    def _update_particles(self):
        """Updates position and lifetime of particles."""
        if not self.particles:
            return
        
        dt = 1/30.0 # Assuming 30fps
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= dt

        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        """Closes the Pygame window."""
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for manual play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Graveyard Survivor")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering for display ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Control the speed of the manual play loop

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a few seconds to show the final message
    pygame.time.wait(3000)
    
    env.close()