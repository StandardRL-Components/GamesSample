import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:05:26.920669
# Source Brief: brief_02571.md
# Brief Index: 2571
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for an arcade spaceship game.
    The player must survive for a set duration by collecting stars to recharge
    shields while dodging and destroying asteroids.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Survive a deadly asteroid field by dodging space rocks and collecting stars to recharge your shields."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move your ship and dodge asteroids."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000  # 5 minutes at 10 steps/second (3000 steps) -> 300 seconds

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (150, 150, 255, 100)
    COLOR_STAR = (255, 255, 0)
    COLOR_STAR_GLOW = (255, 255, 0, 80)
    COLOR_ASTEROID = (128, 132, 142)
    COLOR_ASTEROID_DMG = (255, 100, 100)
    COLOR_EXPLOSION = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SHIELD_HIGH = (0, 255, 128)
    COLOR_SHIELD_MED = (255, 255, 0)
    COLOR_SHIELD_LOW = (255, 0, 0)

    # Player settings
    PLAYER_SPEED = 8
    PLAYER_RADIUS = 12

    # Star settings
    STAR_RADIUS = 6
    MAX_STARS = 5
    STAR_RECHARGE_AMOUNT = 10

    # Asteroid settings
    ASTEROID_MIN_SPEED = 0.5
    ASTEROID_MAX_SPEED = 2.0
    ASTEROID_MIN_SIZE = 10
    ASTEROID_MAX_SIZE = 25
    ASTEROID_MAX_COUNT = 20
    ASTEROID_INITIAL_SPAWN_INTERVAL = 50
    ASTEROID_DIFFICULTY_INTERVAL = 500

    # Explosion settings
    EXPLOSION_RADIUS = 60
    EXPLOSION_DURATION = 15 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.shield = 0
        self.stars = []
        self.asteroids = []
        self.explosions = []
        self.asteroid_spawn_timer = 0
        self.current_asteroid_spawn_interval = 0
        
        # --- Self-Validation ---
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        self.shield = 100.0

        self.stars = []
        for _ in range(self.MAX_STARS):
            self._spawn_star()

        self.asteroids = []
        self.explosions = []
        self.asteroid_spawn_timer = 0
        self.current_asteroid_spawn_interval = self.ASTEROID_INITIAL_SPAWN_INTERVAL
        
        for _ in range(3): # Start with a few asteroids
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        self._handle_player_movement(movement)

        # --- Game Logic Update ---
        self.steps += 1
        reward = 0.1  # Survival reward per step

        self._update_asteroids()
        self._update_explosions()

        reward += self._handle_collisions()

        self._spawn_entities()
        self._update_difficulty()

        # --- Termination Check ---
        terminated = False
        if self.shield <= 0:
            self.game_over = True
            terminated = True
            reward += -100.0
            # sfx: game_over_sound
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            if self.shield >= 50:
                reward += 100.0  # Victory bonus
                # sfx: victory_sound
            else:
                reward += -50.0 # Failed to meet shield requirement

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # Truncated is always False
            self._get_info()
        )

    # --- Update and Logic Methods ---

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'][0] += asteroid['vel'][0]
            asteroid['pos'][1] += asteroid['vel'][1]
            asteroid['angle'] += asteroid['rot_speed']

            # Screen wrap for asteroids
            if asteroid['pos'][0] < -asteroid['size']: asteroid['pos'][0] = self.SCREEN_WIDTH + asteroid['size']
            if asteroid['pos'][0] > self.SCREEN_WIDTH + asteroid['size']: asteroid['pos'][0] = -asteroid['size']
            if asteroid['pos'][1] < -asteroid['size']: asteroid['pos'][1] = self.SCREEN_HEIGHT + asteroid['size']
            if asteroid['pos'][1] > self.SCREEN_HEIGHT + asteroid['size']: asteroid['pos'][1] = -asteroid['size']

    def _update_explosions(self):
        for explosion in self.explosions:
            explosion['life'] -= 1
        self.explosions = [exp for exp in self.explosions if exp['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player-Star collisions
        collected_stars = []
        for star in self.stars:
            dist = math.hypot(self.player_pos[0] - star['pos'][0], self.player_pos[1] - star['pos'][1])
            if dist < self.PLAYER_RADIUS + self.STAR_RADIUS:
                collected_stars.append(star)
                self.shield = min(100.0, self.shield + self.STAR_RECHARGE_AMOUNT)
                reward += 1.0
                # sfx: star_collect_sound

        self.stars = [star for star in self.stars if star not in collected_stars]

        # Asteroid collisions (Player and Explosions)
        asteroids_to_remove = set()
        
        # Player-Asteroid
        for i, asteroid in enumerate(self.asteroids):
            dist = math.hypot(self.player_pos[0] - asteroid['pos'][0], self.player_pos[1] - asteroid['pos'][1])
            if dist < self.PLAYER_RADIUS + asteroid['size']:
                asteroids_to_remove.add(i)
                self.shield -= asteroid['size'] * 0.75  # Larger asteroids do more damage
                self.shield = max(0, self.shield)
                self._create_explosion(asteroid['pos'], asteroid['size'] * 2)
                reward += 5.0 # Reward for destroying an asteroid (even by ramming)
                # sfx: player_hit_sound, explosion_sound
        
        # Explosion-Asteroid (Chain Reaction)
        chain_reaction_count = 0
        for explosion in self.explosions:
            for i, asteroid in enumerate(self.asteroids):
                if i in asteroids_to_remove:
                    continue
                dist = math.hypot(explosion['pos'][0] - asteroid['pos'][0], explosion['pos'][1] - asteroid['pos'][1])
                if dist < explosion['radius'] + asteroid['size']:
                    asteroids_to_remove.add(i)
                    self._create_explosion(asteroid['pos'], asteroid['size'] * 2)
                    chain_reaction_count += 1
                    # sfx: explosion_sound

        if chain_reaction_count > 1:
            reward += 10.0 # Bonus for multi-asteroid chain reaction
        elif chain_reaction_count == 1:
            reward += 5.0 # Standard destruction reward
            
        if asteroids_to_remove:
            self.asteroids = [ast for i, ast in enumerate(self.asteroids) if i not in asteroids_to_remove]
            
        return reward

    def _spawn_entities(self):
        # Respawn stars
        while len(self.stars) < self.MAX_STARS:
            self._spawn_star()
            
        # Spawn new asteroids
        self.asteroid_spawn_timer += 1
        if self.asteroid_spawn_timer >= self.current_asteroid_spawn_interval and len(self.asteroids) < self.ASTEROID_MAX_COUNT:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = 0
    
    def _update_difficulty(self):
        # Increase asteroid spawn rate over time
        difficulty_level = self.steps // self.ASTEROID_DIFFICULTY_INTERVAL
        self.current_asteroid_spawn_interval = max(10, self.ASTEROID_INITIAL_SPAWN_INTERVAL - difficulty_level * 5)

    def _spawn_star(self):
        self.stars.append({
            'pos': [
                self.np_random.integers(self.STAR_RADIUS, self.SCREEN_WIDTH - self.STAR_RADIUS),
                self.np_random.integers(self.STAR_RADIUS, self.SCREEN_HEIGHT - self.STAR_RADIUS)
            ]
        })

    def _spawn_asteroid(self):
        # Spawn on the edge of the screen
        edge = self.np_random.integers(4)
        if edge == 0:  # Top
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ASTEROID_MAX_SIZE]
        elif edge == 1:  # Bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ASTEROID_MAX_SIZE]
        elif edge == 2:  # Left
            pos = [-self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        else:  # Right
            pos = [self.SCREEN_WIDTH + self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        
        angle = math.atan2(self.SCREEN_HEIGHT/2 - pos[1], self.SCREEN_WIDTH/2 - pos[0])
        angle += self.np_random.uniform(-math.pi / 4, math.pi / 4)
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        
        # Generate vertices for an irregular polygon
        num_vertices = self.np_random.integers(5, 9)
        vertices = []
        for i in range(num_vertices):
            angle_vert = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(size * 0.8, size * 1.2)
            vertices.append((math.cos(angle_vert) * dist, math.sin(angle_vert) * dist))
            
        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'size': size,
            'vertices': vertices,
            'angle': self.np_random.uniform(0, 2 * math.pi),
            'rot_speed': self.np_random.uniform(-0.02, 0.02)
        })

    def _create_explosion(self, pos, radius):
        self.explosions.append({
            'pos': list(pos),
            'radius': radius,
            'life': self.EXPLOSION_DURATION
        })

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_stars()
        self._render_explosions()
        self._render_asteroids()
        self._render_player()
        
        if self.game_over:
            self._render_game_over()

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (x - glow_radius, y - glow_radius))

        # Ship body (triangle)
        points = [
            (x, y - self.PLAYER_RADIUS),
            (x - self.PLAYER_RADIUS / 1.5, y + self.PLAYER_RADIUS / 2),
            (x + self.PLAYER_RADIUS / 1.5, y + self.PLAYER_RADIUS / 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_stars(self):
        for star in self.stars:
            x, y = int(star['pos'][0]), int(star['pos'][1])
            
            # Glow
            glow_radius = int(self.STAR_RADIUS * 2)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_STAR_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (x - glow_radius, y - glow_radius))

            # Core star
            pygame.gfxdraw.aacircle(self.screen, x, y, self.STAR_RADIUS, self.COLOR_STAR)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.STAR_RADIUS, self.COLOR_STAR)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = asteroid['pos']
            
            # Check if asteroid was damaged by player this frame
            is_damaged = False
            dist_to_player = math.hypot(self.player_pos[0] - x, self.player_pos[1] - y)
            if dist_to_player < self.PLAYER_RADIUS + asteroid['size']:
                is_damaged = True
            
            color = self.COLOR_ASTEROID_DMG if is_damaged else self.COLOR_ASTEROID
            
            # Rotate vertices
            rotated_vertices = []
            for vx, vy in asteroid['vertices']:
                rot_x = vx * math.cos(asteroid['angle']) - vy * math.sin(asteroid['angle'])
                rot_y = vx * math.sin(asteroid['angle']) + vy * math.cos(asteroid['angle'])
                rotated_vertices.append((int(x + rot_x), int(y + rot_y)))
            
            if len(rotated_vertices) > 2:
                pygame.gfxdraw.aapolygon(self.screen, rotated_vertices, color)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_vertices, color)

    def _render_explosions(self):
        for explosion in self.explosions:
            progress = (self.EXPLOSION_DURATION - explosion['life']) / self.EXPLOSION_DURATION
            current_radius = int(explosion['radius'] * progress)
            alpha = int(255 * (1 - progress))
            if alpha > 0:
                color = (*self.COLOR_EXPLOSION, alpha)
                s = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (current_radius, current_radius), current_radius)
                self.screen.blit(s, (int(explosion['pos'][0] - current_radius), int(explosion['pos'][1] - current_radius)))

    def _render_ui(self):
        # Shield Bar
        shield_percent = self.shield / 100.0
        bar_width = 200
        bar_height = 20
        
        if shield_percent > 0.6:
            shield_color = self.COLOR_SHIELD_HIGH
        elif shield_percent > 0.3:
            shield_color = self.COLOR_SHIELD_MED
        else:
            shield_color = self.COLOR_SHIELD_LOW
            
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, shield_color, (10, 10, int(bar_width * shield_percent), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)
        shield_text = self.font_ui.render(f"SHIELD", True, self.COLOR_UI_TEXT)
        self.screen.blit(shield_text, (15, 11))
        
        # Score and Time
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / (self.metadata['render_fps'] / 3) # approx seconds
        time_text = self.font_ui.render(f"TIME: {int(time_left)}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 30))

    def _render_game_over(self):
        text = "VICTORY" if self.shield > 0 else "GAME OVER"
        color = self.COLOR_SHIELD_HIGH if self.shield > 0 else self.COLOR_SHIELD_LOW
        
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        text_surface = self.font_game_over.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shield": self.shield,
            "asteroids_on_screen": len(self.asteroids)
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
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override pygame screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    
    terminated = False
    total_reward = 0
    
    print("Controls: Arrow keys to move. Close window to quit.")
    
    while not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
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
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Render ---
        # The observation is already rendered to env.screen, so just update display
        pygame.display.flip()
        
        # --- Clock Tick ---
        # The game logic runs at 10Hz (30fps / 3 steps per frame), so we tick at 30fps
        env.clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()