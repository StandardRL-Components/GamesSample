
# Generated: 2025-08-28T05:46:57.080413
# Source Brief: brief_02735.md
# Brief Index: 2735

        
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

    user_guide = (
        "Controls: Press space to jump over the red obstacles. Time your jumps to the beat!"
    )

    game_description = (
        "A minimalist side-scrolling rhythm game. Jump over procedurally generated obstacles to the beat, "
        "creating a satisfying flow state. Reach the end of the level to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG_TOP = (45, 10, 60)
        self.COLOR_BG_BOTTOM = (100, 20, 40)
        self.COLOR_GROUND = (30, 30, 40)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BEAT = (255, 255, 255)
        
        # Physics & Game Rules
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.WORLD_SPEED = 5
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 30
        self.PLAYER_SCREEN_X = self.SCREEN_WIDTH // 4
        self.GROUND_Y = self.SCREEN_HEIGHT - 50
        self.MAX_STEPS = 1000
        self.WIN_DISTANCE = 5000
        self.BPM = 60
        self.BEAT_INTERVAL = self.FPS * 60 // self.BPM
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_lives = pygame.font.Font(None, 48)

        # Pre-render background for performance
        self._create_background_surface()

        # --- State Variables ---
        # These are initialized in reset() but defined here for clarity
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False
        self.lives = 0
        self.world_x_offset = 0
        self.obstacles = []
        self.particles = []
        self.max_obstacle_height = 0
        self.max_obstacle_gap = 0
        self.last_obstacle_x = 0

        self.reset()
        self.validate_implementation()

    def _create_background_surface(self):
        """Creates a pre-rendered gradient background surface."""
        self.background_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background_surface, color, (0, y), (self.SCREEN_WIDTH, y))
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False
        
        self.lives = 3
        self.world_x_offset = 0
        self.obstacles = []
        self.particles = []
        
        # Difficulty settings
        self.max_obstacle_height = 40
        self.max_obstacle_gap = 400
        self.last_obstacle_x = self.SCREEN_WIDTH + 100

        self._generate_initial_obstacles()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_obstacles(self):
        """Fills the screen with initial obstacles on reset."""
        while self.last_obstacle_x < self.SCREEN_WIDTH * 2:
            self._generate_obstacle()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Use rising edge of space press to trigger a single jump
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- Game Logic ---
        self.steps += 1
        
        # Update player
        self._update_player(space_pressed)

        # Update world
        self.world_x_offset += self.WORLD_SPEED
        self._update_obstacles()
        self._update_particles()
        
        # Calculate rewards and check for collisions
        collision_reward, clear_reward = self._calculate_rewards_and_collisions()
        reward += collision_reward + clear_reward
        
        # Continuous penalty for being on the ground
        if self.on_ground:
            reward -= 0.2

        # --- Termination Checks ---
        if self.lives <= 0:
            reward -= 100
            self.game_over = True
        elif self.world_x_offset >= self.WIN_DISTANCE:
            reward += 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, space_pressed):
        """Handles player physics (jump, gravity, ground collision)."""
        if space_pressed and self.on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound.play()
            self._create_particles(15, (self.PLAYER_SCREEN_X + self.PLAYER_WIDTH / 2, self.GROUND_Y), self.COLOR_PLAYER, ((-2, 2), (-5, -1)))

        if not self.on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y - self.PLAYER_HEIGHT:
            if not self.on_ground: # Just landed
                # sfx: land_sound.play()
                self._create_particles(20, (self.PLAYER_SCREEN_X + self.PLAYER_WIDTH / 2, self.GROUND_Y), self.COLOR_GROUND, ((-4, 4), (-2, 0)))
            self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vy = 0
            self.on_ground = True

    def _update_obstacles(self):
        """Scrolls, removes, and generates new obstacles."""
        # Scroll and remove off-screen obstacles
        self.obstacles = [ob for ob in self.obstacles if ob['rect'].x + ob['rect'].width - self.world_x_offset > 0]
        
        # Generate new obstacles on beat if there's space
        if self.steps % self.BEAT_INTERVAL == 0:
            last_ob_screen_x = self.last_obstacle_x - self.world_x_offset
            if last_ob_screen_x < self.SCREEN_WIDTH:
                self._generate_obstacle()
        
        # Increase difficulty every 50 steps
        if self.steps > 0 and self.steps % 50 == 0:
            self.max_obstacle_height = min(150, self.max_obstacle_height + 10)
            self.max_obstacle_gap = max(self.PLAYER_WIDTH + 20, self.max_obstacle_gap - 5)

    def _generate_obstacle(self):
        """Adds a new obstacle to the list."""
        min_gap = self.PLAYER_WIDTH + 80
        gap = self.np_random.integers(min_gap, self.max_obstacle_gap + 1)
        
        height = self.np_random.integers(20, self.max_obstacle_height + 1)
        width = self.np_random.integers(30, 60)
        
        x_pos = self.last_obstacle_x + gap
        y_pos = self.GROUND_Y - height
        
        new_obstacle = {
            'rect': pygame.Rect(x_pos, y_pos, width, height),
            'cleared': False,
            'is_tall': height > 100
        }
        self.obstacles.append(new_obstacle)
        self.last_obstacle_x = x_pos + width

    def _calculate_rewards_and_collisions(self):
        """Checks for player-obstacle collisions and successful clears."""
        collision_reward = 0
        clear_reward = 0
        player_rect = pygame.Rect(self.PLAYER_SCREEN_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for ob in self.obstacles:
            ob_rect = ob['rect']
            ob_screen_x = ob_rect.x - self.world_x_offset
            ob_screen_rect = pygame.Rect(ob_screen_x, ob_rect.y, ob_rect.width, ob_rect.height)

            # Check for collision
            if player_rect.colliderect(ob_screen_rect):
                if self.lives > 0: # Only penalize on the frame of impact
                    collision_reward -= 5
                    self.lives -= 1
                    # sfx: collision_sound.play()
                    self._create_particles(30, player_rect.center, self.COLOR_OBSTACLE, ((-5, 5), (-5, 5)))
                    # Mark as cleared to prevent further interaction
                    ob['cleared'] = True 
                break # Stop after one collision per frame
            
            # Check for successful clear
            if not ob['cleared'] and player_rect.left > ob_screen_rect.right:
                clear_reward += 5 if ob['is_tall'] else 1
                ob['cleared'] = True
                # sfx: clear_sound.play()

        return collision_reward, clear_reward
        
    def _update_particles(self):
        """Updates position and lifespan of all particles."""
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _create_particles(self, count, pos, color, vel_range):
        """Factory for creating a burst of particles."""
        for _ in range(count):
            particle = {
                'pos': list(pos),
                'vel': [self.np_random.uniform(vel_range[0][0], vel_range[0][1]), 
                        self.np_random.uniform(vel_range[1][0], vel_range[1][1])],
                'lifespan': self.np_random.integers(10, 25),
                'color': color
            }
            self.particles.append(particle)

    def _get_observation(self):
        # --- Render all game elements to the surface ---
        self.screen.blit(self.background_surface, (0, 0))
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        
        # Beat indicator
        beat_progress = (self.steps % self.BEAT_INTERVAL) / self.BEAT_INTERVAL
        pulse = abs(math.sin(beat_progress * math.pi))
        beat_radius = int(15 + pulse * 10)
        beat_alpha = int(50 + pulse * 100)
        beat_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25)
        pygame.gfxdraw.filled_circle(self.screen, beat_pos[0], beat_pos[1], beat_radius, (*self.COLOR_BEAT, beat_alpha))
        pygame.gfxdraw.aacircle(self.screen, beat_pos[0], beat_pos[1], beat_radius, (*self.COLOR_BEAT, beat_alpha))

        # Obstacles
        for ob in self.obstacles:
            ob_rect = ob['rect']
            ob_screen_x = ob_rect.x - self.world_x_offset
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (ob_screen_x, ob_rect.y, ob_rect.width, ob_rect.height))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 25))))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, p['pos'], 2)

        # Player
        player_rect = pygame.Rect(self.PLAYER_SCREEN_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), glow_surface.get_rect(), border_radius=4)
        self.screen.blit(glow_surface, glow_rect.topleft)
        # Main player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        """Draws score and lives on the screen."""
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_lives.render("♥" * self.lives, True, self.COLOR_OBSTACLE)
        text_rect = lives_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(lives_text, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "distance": self.world_x_offset,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Rhythm Jumper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # no-op
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held
        # Other actions are unused in this game
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()