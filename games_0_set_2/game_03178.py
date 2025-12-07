
# Generated: 2025-08-27T22:35:56.782467
# Source Brief: brief_03178.md
# Brief Index: 3178

        
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
        "Controls: Arrow keys to move the cursor. Hold Space or Shift to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice the falling fruit to score points, but be careful to avoid the bombs!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Gameplay settings
    WIN_SCORE = 30
    MAX_LIVES = 3
    CURSOR_SPEED = 15.0
    SLICE_RADIUS = 25
    
    # Initial difficulty
    INITIAL_SPAWN_RATE = 40 # Lower is faster
    INITIAL_OBJECT_SPEED = 2.0
    
    # Difficulty scaling
    DIFFICULTY_INTERVAL = 150 # steps
    SPAWN_RATE_DECREASE = 2
    SPEED_INCREASE = 0.25

    # Colors
    COLOR_BG_TOP = (15, 20, 30)
    COLOR_BG_BOTTOM = (35, 40, 60)
    COLOR_CURSOR = (200, 255, 255)
    COLOR_CURSOR_GLOW = (100, 150, 150)
    COLOR_SLICE_TRAIL = (255, 255, 255)
    COLOR_BOMB = (20, 20, 20)
    COLOR_BOMB_SKULL = (200, 200, 200)
    COLOR_TEXT = (220, 220, 220)

    FRUIT_PROPS = {
        'apple': {'color': (220, 30, 30), 'radius': 18},
        'orange': {'color': (250, 150, 20), 'radius': 20},
        'kiwi': {'color': (100, 200, 50), 'radius': 16},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Internal state variables are initialized in reset()
        self.cursor_pos = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.slice_trail = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.spawn_timer = None
        self.current_spawn_rate = None
        self.current_object_speed = None
        
        # Create a pre-rendered background for efficiency
        self.background = self._create_background()
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trail = []

        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.current_object_speed = self.INITIAL_OBJECT_SPEED
        self.spawn_timer = self.current_spawn_rate
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, do nothing
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        step_reward = 0

        # --- 1. Unpack Action and Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        is_slicing = space_held or shift_held
        
        self._update_cursor(movement)
        self._update_slice_trail(is_slicing)

        # --- 2. Update Game Logic ---
        self._update_objects()
        self._update_particles()
        self._spawn_objects()

        if is_slicing:
            # Sound effect placeholder: # sfx_slice_swing.play()
            step_reward += self._handle_slicing()

        # --- 3. Calculate Reward and Termination ---
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if self.score >= self.WIN_SCORE:
                step_reward += 100 # Win bonus
            elif self.lives <= 0:
                step_reward -= 100 # Lose penalty
        
        # --- 4. Update Difficulty ---
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_object_speed += self.SPEED_INCREASE
            self.current_spawn_rate = max(10, self.current_spawn_rate - self.SPAWN_RATE_DECREASE)

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_cursor(self, movement):
        """Updates cursor position based on movement action."""
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _update_slice_trail(self, is_slicing):
        """Manages the visual trail of the slice."""
        # Add new point if slicing
        if is_slicing:
            self.slice_trail.append({'pos': self.cursor_pos.copy(), 'life': 10})
        
        # Update and remove old points
        new_trail = []
        for point in self.slice_trail:
            point['life'] -= 1
            if point['life'] > 0:
                new_trail.append(point)
        self.slice_trail = new_trail

    def _spawn_objects(self):
        """Spawns new fruits and bombs."""
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.current_spawn_rate
            
            pos = np.array([random.uniform(self.WIDTH * 0.1, self.WIDTH * 0.9), -30.0], dtype=np.float32)
            angle = random.uniform(-0.5, 0.5)
            speed = self.current_object_speed + random.uniform(-0.5, 0.5)
            vel = np.array([math.sin(angle) * speed, math.cos(angle) * speed], dtype=np.float32)

            if random.random() > 0.2: # 80% chance for fruit
                fruit_type = random.choice(list(self.FRUIT_PROPS.keys()))
                props = self.FRUIT_PROPS[fruit_type]
                self.fruits.append({'pos': pos, 'vel': vel, 'radius': props['radius'], 'color': props['color']})
            else: # 20% chance for bomb
                self.bombs.append({'pos': pos, 'vel': vel, 'radius': 22})

    def _update_objects(self):
        """Moves all falling objects and removes those off-screen."""
        for fruit in self.fruits[:]:
            fruit['pos'] += fruit['vel']
            if fruit['pos'][1] > self.HEIGHT + fruit['radius']:
                self.fruits.remove(fruit)
        
        for bomb in self.bombs[:]:
            bomb['pos'] += bomb['vel']
            if bomb['pos'][1] > self.HEIGHT + bomb['radius']:
                self.bombs.remove(bomb)

    def _handle_slicing(self):
        """Checks for slice collisions and processes results."""
        reward = 0
        
        # Check for fruit slices
        for fruit in self.fruits[:]:
            dist = np.linalg.norm(self.cursor_pos - fruit['pos'])
            if dist < fruit['radius'] + self.SLICE_RADIUS:
                # Sound effect placeholder: # sfx_fruit_slice.play()
                self._create_fruit_particles(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)
                self.score += 1
                reward += 1

        # Check for bomb slices
        for bomb in self.bombs[:]:
            dist = np.linalg.norm(self.cursor_pos - bomb['pos'])
            if dist < bomb['radius'] + self.SLICE_RADIUS:
                # Sound effect placeholder: # sfx_bomb_explode.play()
                self._create_explosion_particles(bomb['pos'])
                self.bombs.remove(bomb)
                self.lives -= 1
                reward -= 5

        return reward

    def _create_fruit_particles(self, pos, color):
        """Creates particles for a sliced fruit."""
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            life = random.randint(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color, 'radius': random.uniform(2, 4)})

    def _create_explosion_particles(self, pos):
        """Creates particles for a bomb explosion."""
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 7)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            life = random.randint(20, 40)
            color = random.choice([(255, 50, 50), (255, 150, 50), (255, 255, 100)])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color, 'radius': random.uniform(2, 5)})

    def _update_particles(self):
        """Updates position and lifespan of all particles."""
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _check_termination(self):
        """Checks for win, loss, or timeout conditions."""
        if self.score >= self.WIN_SCORE or self.lives <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        """Renders the game state to the screen and returns it as a numpy array."""
        # --- Render all game elements ---
        self.screen.blit(self.background, (0, 0))
        self._render_particles()
        self._render_objects()
        self._render_slice_trail()
        self._render_cursor()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with current game information."""
        return {
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
        }

    def _create_background(self):
        """Creates a pre-rendered gradient background surface."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_objects(self):
        """Renders all fruits and bombs."""
        for fruit in self.fruits:
            pos = fruit['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
        
        for bomb in self.bombs:
            pos = bomb['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            # Draw a simple skull
            skull_pos = (pos[0], pos[1] + 2)
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (skull_pos[0]-6, skull_pos[1]-4, 12, 8))
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (skull_pos[0]-2, skull_pos[1]+4, 4, 4))
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (skull_pos[0]-4, skull_pos[1]-1), 2)
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (skull_pos[0]+4, skull_pos[1]-1), 2)


    def _render_particles(self):
        """Renders all active particles."""
        for p in self.particles:
            pos = p['pos'].astype(int)
            alpha = int(255 * (p['life'] / 30))
            alpha = max(0, min(255, alpha))
            color = (*p['color'], alpha)
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))


    def _render_slice_trail(self):
        """Renders the player's slice trail."""
        if len(self.slice_trail) > 1:
            points = [p['pos'] for p in self.slice_trail]
            # This is a simplified trail; for true fading, one would draw segments with decreasing alpha.
            # aalines is performant and looks good enough.
            pygame.draw.aalines(self.screen, self.COLOR_SLICE_TRAIL, False, points, 2)


    def _render_cursor(self):
        """Renders the player's cursor with a glow effect."""
        pos = self.cursor_pos.astype(int)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLICE_RADIUS, self.COLOR_CURSOR_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLICE_RADIUS, self.COLOR_CURSOR_GLOW)
        # Main cursor
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_CURSOR)

    def _render_ui(self):
        """Renders the score and lives."""
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Lives (bombs)
        for i in range(self.MAX_LIVES):
            pos = (self.WIDTH - 40 - i * 40, 30)
            color = self.COLOR_BOMB if i < self.lives else (80, 20, 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, color)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # Use a dictionary to track held keys for smooth movement
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False,
    }

    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False
        
        # --- Action Mapping ---
        movement = 0 # No-op
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 1 if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(env.FPS)

    env.close()