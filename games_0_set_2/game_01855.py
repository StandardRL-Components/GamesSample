import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to dummy to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to turn your ship. Press space to fire your weapon. "
        "Survive and destroy all asteroids!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down arcade shooter. Pilot your ship, blast asteroids for points, "
        "and build up chain bonuses for a high score. Avoid collisions or you'll lose a life!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    INITIAL_ASTEROIDS = 20
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_LASER = (255, 50, 50)
    COLOR_ASTEROID = (200, 200, 200)
    COLOR_EXPLOSION_1 = (255, 200, 50)
    COLOR_EXPLOSION_2 = (255, 100, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_BOUNDARY = (100, 100, 120)

    # Player Physics
    PLAYER_ROTATION_SPEED = 5  # degrees per step
    PLAYER_RADIUS = 12
    INVULNERABILITY_DURATION = 90 # steps

    # Laser Physics
    LASER_SPEED = 10
    LASER_COOLDOWN = 8 # steps

    # Asteroid Physics
    ASTEROID_MIN_SPEED = 0.5
    ASTEROID_MAX_SPEED = 1.5
    ASTEROID_MIN_RADIUS = 10
    ASTEROID_MAX_RADIUS = 30
    ASTEROID_MIN_VERTICES = 5
    ASTEROID_MAX_VERTICES = 12

    # Scoring & Rewards
    REWARD_SURVIVAL = 0.01
    REWARD_ASTEROID_DESTROYED = 1.0
    REWARD_CHAIN_BONUS = 0.5
    REWARD_MISSED_SHOT = -0.2
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    CHAIN_TIMEOUT = 90 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables to be defined in reset()
        self.player = {}
        self.asteroids = []
        self.lasers = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laser_cooldown_timer = 0
        self.chain_multiplier = 1
        self.chain_timer = 0
        
        # This call is for internal validation, it's not part of the standard API
        # but is useful for ensuring the environment is correctly implemented.
        # It calls reset() itself to initialize the state.
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.chain_multiplier = 1
        self.chain_timer = 0

        self.player = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float64),
            "angle": -90.0,
            "lives": self.INITIAL_LIVES,
            "invulnerability": self.INVULNERABILITY_DURATION
        }

        self.lasers = []
        self.explosions = []
        self.asteroids = self._spawn_asteroids()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = self.REWARD_SURVIVAL
        self.steps += 1

        # -- Handle Actions --
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3: # Left
            self.player['angle'] -= self.PLAYER_ROTATION_SPEED
        elif movement == 4: # Right
            self.player['angle'] += self.PLAYER_ROTATION_SPEED
        
        if self.laser_cooldown_timer > 0:
            self.laser_cooldown_timer -= 1

        if space_pressed and self.laser_cooldown_timer == 0:
            self._fire_laser()
            reward += self.REWARD_MISSED_SHOT # Penalize firing, reward is added back on hit

        # -- Update Game State --
        self._update_timers()
        self._update_lasers()
        self._update_asteroids()
        self._update_explosions()

        # -- Handle Collisions --
        reward += self._handle_laser_asteroid_collisions()
        self._handle_player_asteroid_collision()
        self._handle_asteroid_asteroid_collisions()

        # -- Check Termination Conditions --
        terminated = False
        if self.player['lives'] <= 0:
            reward = self.REWARD_LOSS
            terminated = True
        elif not self.asteroids:
            reward = self.REWARD_WIN
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per new Gym API, if truncated, terminated should be true
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player.get('lives', 0),
            "asteroids_remaining": len(self.asteroids),
        }

    # --- Helper methods for game logic ---

    def _spawn_asteroids(self):
        asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            while True:
                radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
                pos = self.np_random.uniform([radius, radius], [self.SCREEN_WIDTH - radius, self.SCREEN_HEIGHT - radius])
                
                # Ensure it doesn't spawn too close to the player's start
                if np.linalg.norm(pos - self.player['pos']) < radius + self.PLAYER_RADIUS + 50:
                    continue
                
                # Ensure it doesn't overlap with other asteroids
                overlap = False
                for other in asteroids:
                    if np.linalg.norm(pos - other['pos']) < radius + other['radius']:
                        overlap = True
                        break
                if not overlap:
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            num_vertices = self.np_random.integers(self.ASTEROID_MIN_VERTICES, self.ASTEROID_MAX_VERTICES + 1)
            # Generate jagged points for the polygon
            point_offsets = self.np_random.uniform(0.7, 1.3, num_vertices)

            asteroids.append({
                "pos": pos,
                "velocity": velocity,
                "radius": radius,
                "num_vertices": num_vertices,
                "point_offsets": point_offsets
            })
        return asteroids

    def _fire_laser(self):
        # sfx: player_shoot.wav
        self.laser_cooldown_timer = self.LASER_COOLDOWN
        angle_rad = math.radians(self.player['angle'])
        start_pos = self.player['pos'] + np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.PLAYER_RADIUS
        velocity = np.array([math.cos(angle_rad) * self.LASER_SPEED, math.sin(angle_rad) * self.LASER_SPEED])
        self.lasers.append({"pos": start_pos, "velocity": velocity})

    def _update_timers(self):
        if self.player.get('invulnerability', 0) > 0:
            self.player['invulnerability'] -= 1
        
        if self.chain_timer > 0:
            self.chain_timer -= 1
        else:
            self.chain_multiplier = 1

    def _update_lasers(self):
        self.lasers = [
            laser for laser in self.lasers
            if 0 < laser['pos'][0] < self.SCREEN_WIDTH and 0 < laser['pos'][1] < self.SCREEN_HEIGHT
        ]
        for laser in self.lasers:
            laser['pos'] += laser['velocity']

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['velocity']
            # Wall bouncing
            if asteroid['pos'][0] < asteroid['radius'] or asteroid['pos'][0] > self.SCREEN_WIDTH - asteroid['radius']:
                asteroid['velocity'][0] *= -1
                asteroid['pos'][0] = np.clip(asteroid['pos'][0], asteroid['radius'], self.SCREEN_WIDTH - asteroid['radius'])
            if asteroid['pos'][1] < asteroid['radius'] or asteroid['pos'][1] > self.SCREEN_HEIGHT - asteroid['radius']:
                asteroid['velocity'][1] *= -1
                asteroid['pos'][1] = np.clip(asteroid['pos'][1], asteroid['radius'], self.SCREEN_HEIGHT - asteroid['radius'])

    def _update_explosions(self):
        for explosion in self.explosions:
            explosion['progress'] += explosion['speed']
        self.explosions = [e for e in self.explosions if e['progress'] < 1.0]

    def _handle_laser_asteroid_collisions(self):
        hit_reward = 0
        lasers_to_keep = []
        asteroids_to_remove = set()

        for i, laser in enumerate(self.lasers):
            hit = False
            for j, asteroid in enumerate(self.asteroids):
                if j in asteroids_to_remove:
                    continue
                dist = np.linalg.norm(laser['pos'] - asteroid['pos'])
                if dist < asteroid['radius']:
                    asteroids_to_remove.add(j)
                    hit = True
                    break # Laser can only hit one asteroid
            if not hit:
                lasers_to_keep.append(laser)

        if asteroids_to_remove:
            # sfx: explosion.wav
            remaining_asteroids = []
            for i, asteroid in enumerate(self.asteroids):
                if i in asteroids_to_remove:
                    self._create_explosion(asteroid['pos'], asteroid['radius'])
                    
                    # Calculate reward for this hit
                    hit_reward += self.REWARD_ASTEROID_DESTROYED
                    hit_reward += self.REWARD_CHAIN_BONUS * self.chain_multiplier
                    hit_reward -= self.REWARD_MISSED_SHOT # Negate the initial penalty
                    
                    self.score += 10 * self.chain_multiplier
                    self.chain_multiplier += 1
                    self.chain_timer = self.CHAIN_TIMEOUT
                else:
                    remaining_asteroids.append(asteroid)
            self.asteroids = remaining_asteroids
        
        self.lasers = lasers_to_keep
        return hit_reward

    def _handle_player_asteroid_collision(self):
        if self.player.get('invulnerability', 0) > 0:
            return

        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player['pos'] - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                # sfx: player_hit.wav
                self.player['lives'] -= 1
                self.player['invulnerability'] = self.INVULNERABILITY_DURATION
                self.chain_multiplier = 1
                self.chain_timer = 0
                self._create_explosion(self.player['pos'], self.PLAYER_RADIUS * 2)
                # Respawn player in center, could be changed later
                self.player['pos'] = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float64)
                break

    def _handle_asteroid_asteroid_collisions(self):
        for i in range(len(self.asteroids)):
            for j in range(i + 1, len(self.asteroids)):
                a1 = self.asteroids[i]
                a2 = self.asteroids[j]
                
                dist = np.linalg.norm(a1['pos'] - a2['pos'])
                if dist < a1['radius'] + a2['radius']:
                    # Simple elastic collision: swap velocities
                    # This is not physically accurate but is great for arcade feel
                    v1_new = a2['velocity']
                    v2_new = a1['velocity']
                    a1['velocity'] = v1_new
                    a2['velocity'] = v2_new

                    # Prevent sticking by moving them apart
                    overlap = (a1['radius'] + a2['radius'] - dist)
                    if dist > 1e-6:
                        direction = (a1['pos'] - a2['pos']) / dist
                        a1['pos'] += direction * overlap / 2
                        a2['pos'] -= direction * overlap / 2

    def _create_explosion(self, pos, size):
        self.explosions.append({'pos': pos, 'size': size, 'progress': 0.0, 'speed': 0.03})

    # --- Helper methods for rendering ---

    def _render_game(self):
        # Arena boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

        # Asteroids
        for asteroid in self.asteroids:
            self._draw_poly(
                self.screen, self.COLOR_ASTEROID, asteroid['pos'], asteroid['radius'],
                asteroid['num_vertices'], 0, asteroid['point_offsets']
            )

        # Lasers
        for laser in self.lasers:
            start_pos = laser['pos']
            end_pos = laser['pos'] - laser['velocity'] * 0.5 # Create a short tail
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos.astype(int), end_pos.astype(int), 2)

        # Player
        if self.player: # Ensure player dict is not empty
            is_invulnerable = self.player.get('invulnerability', 0) > 0
            is_blinking = is_invulnerable and (self.steps // 4 % 2) == 0
            if not is_blinking:
                self._draw_poly(
                    self.screen, self.COLOR_PLAYER, self.player['pos'], self.PLAYER_RADIUS,
                    3, self.player['angle']
                )

        # Explosions
        for explosion in self.explosions:
            progress = explosion['progress']
            current_radius = int(explosion['size'] * math.sin(progress * math.pi / 2))
            alpha = int(255 * (1 - progress))
            if current_radius > 0 and alpha > 0:
                # Draw two circles for a better effect
                color1 = (*self.COLOR_EXPLOSION_1, alpha)
                color2 = (*self.COLOR_EXPLOSION_2, alpha)
                pygame.gfxdraw.aacircle(self.screen, int(explosion['pos'][0]), int(explosion['pos'][1]), current_radius, color1)
                if current_radius > 4:
                    pygame.gfxdraw.aacircle(self.screen, int(explosion['pos'][0]), int(explosion['pos'][1]), max(0, current_radius - 4), color2)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives = self.player.get('lives', 0)
        for i in range(lives):
            ship_icon_pos = np.array([self.SCREEN_WIDTH - 20 - i * 25, 20])
            self._draw_poly(self.screen, self.COLOR_PLAYER, ship_icon_pos, 8, 3, -90)
        
        # Chain Multiplier
        if self.chain_multiplier > 1:
            chain_text = self.font_large.render(f"x{self.chain_multiplier}", True, self.COLOR_EXPLOSION_1)
            text_rect = chain_text.get_rect(center=(self.SCREEN_WIDTH / 2, 40))
            self.screen.blit(chain_text, text_rect)

        # Game Over
        if self.game_over:
            msg = "YOU WIN!" if not self.asteroids else "GAME OVER"
            color = self.COLOR_PLAYER if not self.asteroids else self.COLOR_LASER
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _draw_poly(self, surface, color, center, radius, num_points, rotation, offsets=None):
        points = []
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i + math.radians(rotation)
            offset_radius = radius * (offsets[i] if offsets is not None and i < len(offsets) else 1)
            x = center[0] + offset_radius * math.cos(angle)
            y = center[1] + offset_radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing
    # and see the game being played.
    import sys

    # To run headlessly, uncomment the line below
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # To run with a display, ensure the dummy driver is NOT set
    # and that you have a display environment (e.g., a desktop or X11 forwarding)
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen_width, screen_height = env.SCREEN_WIDTH, env.SCREEN_HEIGHT
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Asteroid Arena")

    terminated = False
    truncated = False
    total_reward = 0
    
    # --- Manual Control ---
    # [movement, space, shift]
    # movement: 0=none, 1=up(unused), 2=down(unused), 3=left, 4=right
    action = np.array([0, 0, 0])

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement (rotation)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Firing
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # No shift action in this game
        # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        #     action[2] = 1

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a few seconds to see the final screen
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()