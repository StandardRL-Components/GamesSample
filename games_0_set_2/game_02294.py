
# Generated: 2025-08-27T19:54:57.770790
# Source Brief: brief_02294.md
# Brief Index: 2294

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    A top-down arcade survival game where the player must avoid a growing horde of zombies for 60 seconds.
    The game is presented in a circular arena with a retro visual style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your green square and avoid the red zombies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds in a top-down arena against a horde of relentless zombies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """Initializes the game environment."""
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_ARENA = (80, 80, 90)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_WARN = (255, 180, 0)

        # Game entities
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 2.0
        self.ZOMBIE_SIZE = 10
        self.INITIAL_ZOMBIE_COUNT = 5
        self.INITIAL_ZOMBIE_SPEED = 0.75
        self.ZOMBIE_SPEED_INCREASE = 0.05
        self.ZOMBIE_SPEED_INCREASE_INTERVAL = self.FPS * 10 # Every 10 seconds

        # Arena
        self.ARENA_CENTER = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.ARENA_RADIUS = 180

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = None
        self.zombies = []
        self.zombie_speed = 0.0
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = self.ARENA_CENTER.copy()
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.particles = []

        # Spawn zombies
        self.zombies = []
        for _ in range(self.INITIAL_ZOMBIE_COUNT):
            self._spawn_zombie()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """Advances the game by one time step."""
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_zombies()
        self._update_particles()
        
        # --- Check Game State ---
        terminated = self._check_collisions()
        win = not terminated and self.steps >= self.MAX_STEPS -1

        if terminated or win:
            self.game_over = True
            if terminated:
                 # sfx: player_death_explosion
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)

        # --- Calculate Reward ---
        reward = 0.1  # Survival reward
        if win:
            # sfx: victory_fanfare
            reward += 100.0  # Win bonus

        # --- Update Counters ---
        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated is always False
            self._get_info(),
        )

    def _handle_input(self, action):
        """Processes the player's action."""
        movement = action[0]
        direction = np.zeros(2, dtype=np.float32)

        if movement == 1:  # Up
            direction[1] = -1
        elif movement == 2:  # Down
            direction[1] = 1
        elif movement == 3:  # Left
            direction[0] = -1
        elif movement == 4:  # Right
            direction[0] = 1

        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        self.player_pos += direction * self.PLAYER_SPEED

        # Clamp player to arena
        dist_from_center = np.linalg.norm(self.player_pos - self.ARENA_CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.PLAYER_SIZE / 2:
            vec_to_player = self.player_pos - self.ARENA_CENTER
            self.player_pos = self.ARENA_CENTER + vec_to_player / dist_from_center * (self.ARENA_RADIUS - self.PLAYER_SIZE / 2)

    def _update_zombies(self):
        """Updates zombie positions and difficulty."""
        # Increase difficulty over time
        if self.steps > 0 and self.steps % self.ZOMBIE_SPEED_INCREASE_INTERVAL == 0:
            self.zombie_speed += self.ZOMBIE_SPEED_INCREASE
            # sfx: difficulty_increase_chime

        for zombie in self.zombies:
            direction = self.player_pos - zombie['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero
                direction /= dist
            zombie['pos'] += direction * self.zombie_speed

    def _check_collisions(self):
        """Checks for collisions between the player and zombies."""
        collision_dist = (self.PLAYER_SIZE + self.ZOMBIE_SIZE) / 2
        for zombie in self.zombies:
            if np.linalg.norm(self.player_pos - zombie['pos']) < collision_dist:
                return True
        return False

    def _spawn_zombie(self):
        """Spawns a single zombie at a random location within the arena."""
        while True:
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(self.ARENA_RADIUS * 0.5, self.ARENA_RADIUS * 0.95)
            pos = self.ARENA_CENTER + np.array([math.cos(angle), math.sin(angle)]) * radius
            if np.linalg.norm(pos - self.player_pos) > 100: # Don't spawn on player
                self.zombies.append({'pos': pos.astype(np.float32)})
                break
    
    def _create_explosion(self, position, num_particles, color):
        """Creates a particle explosion effect."""
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': position.copy(),
                'vel': velocity,
                'life': lifetime,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        """Updates the state of all active particles."""
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # Drag
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the core game elements (arena, player, zombies)."""
        # Arena
        pygame.gfxdraw.filled_circle(self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]), self.ARENA_RADIUS, self.COLOR_ARENA)
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]), self.ARENA_RADIUS, self.COLOR_ARENA)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))
        
        # Zombies
        for z in self.zombies:
            rect = pygame.Rect(0, 0, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            rect.center = (int(z['pos'][0]), int(z['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)

        # Player
        if not (self.game_over and self.steps < self.MAX_STEPS):
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            # Add a small glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE*2, self.PLAYER_SIZE*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 60), (self.PLAYER_SIZE, self.PLAYER_SIZE), self.PLAYER_SIZE)
            self.screen.blit(glow_surf, (player_rect.x - self.PLAYER_SIZE/2, player_rect.y - self.PLAYER_SIZE/2))


    def _render_ui(self):
        """Renders the user interface (timer, score)."""
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_TEXT
        timer_surface = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(timer_surface, (20, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surface = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_surface, score_rect)
        
        # Game Over / Win Message
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "YOU SURVIVED!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ZOMBIE
            
            msg_surface = self.font_large.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surface, msg_rect)


    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "zombie_count": len(self.zombies),
            "zombie_speed": self.zombie_speed
        }

    def close(self):
        """Cleans up the environment's resources."""
        pygame.font.quit()
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Use a different screen for display that can be scaled
    pygame.display.set_caption("Zombie Survival")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH * 1.5, env.SCREEN_HEIGHT * 1.5))
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([movement, space_held, shift_held])
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf, display_screen.get_size())
        display_screen.blit(scaled_surf, (0, 0))
        
        pygame.display.flip()
        
        # If game is over, wait for a key press to reset
        if terminated:
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        waiting_for_reset = False
                env.clock.tick(15) # Don't burn CPU while waiting
            
            if running: # Don't reset if we're quitting
                obs, info = env.reset()
                terminated = False

        env.clock.tick(env.FPS)

    env.close()