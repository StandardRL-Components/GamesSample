import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the crosshair. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-view target practice game. Hit all targets with your limited ammunition. "
        "More distant targets are worth more."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Game constants
        self.MAX_STEPS = 1000
        self.INITIAL_AMMO = 15
        self.NUM_TARGETS = 20
        self.CROSSHAIR_SPEED = 12
        self.PROJECTILE_SPEED = 25
        self.TARGET_RADIUS = 15
        self.GUN_POS = np.array([self.WIDTH / 2, self.HEIGHT - 10], dtype=float)

        # Colors
        self.COLOR_BG = (240, 240, 245)
        self.COLOR_FLOOR = (200, 200, 210)
        self.COLOR_CROSSHAIR = (10, 10, 10)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_TARGET_INNER = (255, 255, 255)
        self.COLOR_PROJECTILE = (255, 200, 0)
        self.COLOR_TEXT = (20, 20, 20)
        self.COLOR_GUN = (80, 80, 90)

        # Initialize state variables
        self.crosshair_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.ammo = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False

        # self.validate_implementation() # This is a helper, not part of the core API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = self.INITIAL_AMMO
        self.prev_space_held = False

        self.crosshair_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)

        self.targets = []
        self.projectiles = []
        self.particles = []

        # Generate targets
        for _ in range(self.NUM_TARGETS):
            # Ensure targets are not too close to each other or edges
            while True:
                pos = np.array([
                    self.np_random.uniform(low=50, high=self.WIDTH - 50),
                    self.np_random.uniform(low=50, high=self.HEIGHT - 150)
                ])
                if not any(np.linalg.norm(pos - t['pos']) < self.TARGET_RADIUS * 3 for t in self.targets):
                    self.targets.append({'pos': pos, 'radius': self.TARGET_RADIUS, 'hit': False})
                    break

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is not used in this game

        # --- UPDATE GAME LOGIC ---

        # 1. Handle crosshair movement
        if movement == 1:  # Up
            self.crosshair_pos[1] -= self.CROSSHAIR_SPEED
        elif movement == 2:  # Down
            self.crosshair_pos[1] += self.CROSSHAIR_SPEED
        elif movement == 3:  # Left
            self.crosshair_pos[0] -= self.CROSSHAIR_SPEED
        elif movement == 4:  # Right
            self.crosshair_pos[0] += self.CROSSHAIR_SPEED

        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.HEIGHT)

        # 2. Handle shooting (on key press, not hold)
        if space_held and not self.prev_space_held and self.ammo > 0 and not self.game_over:
            self.ammo -= 1
            # // Sound: Pew!

            direction = self.crosshair_pos - self.GUN_POS
            norm = np.linalg.norm(direction)
            if norm > 0:
                velocity = (direction / norm) * self.PROJECTILE_SPEED
                self.projectiles.append({'pos': self.GUN_POS.copy(), 'vel': velocity})
        self.prev_space_held = space_held

        # 3. Update projectiles and check for hits
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'] += p['vel']

            # Check for target hits
            hit_target = False
            for t in self.targets:
                if not t['hit'] and np.linalg.norm(p['pos'] - t['pos']) < t['radius']:
                    t['hit'] = True
                    self.score += 1
                    hit_target = True
                    # // Sound: Target Hit!

                    # Base reward for hit
                    reward += 10

                    # Bonus reward for distance from center
                    dist_from_center = np.linalg.norm(t['pos'] - np.array([self.WIDTH / 2, self.HEIGHT / 2]))
                    max_dist = np.linalg.norm(np.array([self.WIDTH / 2, self.HEIGHT / 2]))
                    reward += 5 * (dist_from_center / max_dist)  # Scaled bonus up to +5

                    self._create_explosion(t['pos'], self.COLOR_TARGET)
                    projectiles_to_remove.append(i)
                    break

            # Check if projectile is off-screen
            if not hit_target and not (0 <= p['pos'][0] <= self.WIDTH and 0 <= p['pos'][1] <= self.HEIGHT):
                projectiles_to_remove.append(i)
                # Penalty for missing a shot
                reward -= 0.1

        # Remove projectiles that hit or went off-screen
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]

        # 4. Update particles
        particles_to_remove = []
        for i, part in enumerate(self.particles):
            part['pos'] += part['vel']
            part['lifespan'] -= 1
            if part['lifespan'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

        # 5. Check for termination conditions
        self.steps += 1
        terminated = False
        truncated = False

        all_targets_hit = all(t['hit'] for t in self.targets)
        out_of_ammo = self.ammo <= 0 and not self.projectiles

        if all_targets_hit:
            reward += 100
            terminated = True
            self.game_over = True
        elif out_of_ammo:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.HEIGHT - 40, self.WIDTH, 40))

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw target stands (decorative)
        for t in self.targets:
            if not t['hit']:
                start_pos = (int(t['pos'][0]), int(t['pos'][1] + t['radius']))
                end_pos = (int(t['pos'][0]), self.HEIGHT - 40)
                pygame.draw.line(self.screen, (180, 180, 190), start_pos, end_pos, 2)

        # Draw targets
        for t in self.targets:
            if not t['hit']:
                pos_int = (int(t['pos'][0]), int(t['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], t['radius'], self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], t['radius'], self.COLOR_TARGET)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], t['radius'] // 2,
                                            self.COLOR_TARGET_INNER)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], t['radius'] // 2,
                                        self.COLOR_TARGET_INNER)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 25.0))
            size = int(max(1, 5 * (p['lifespan'] / 25.0)))
            r, g, b = p['color']
            # Simple rect for particles is faster and fine for small sizes
            pygame.draw.rect(self.screen, (r, g, b), (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Draw gun
        pygame.draw.rect(self.screen, self.COLOR_GUN, (self.GUN_POS[0] - 15, self.GUN_POS[1] - 10, 30, 20))
        gun_direction = self.crosshair_pos - self.GUN_POS
        norm = np.linalg.norm(gun_direction)
        if norm > 0:
            end_pos = self.GUN_POS + (gun_direction / norm) * 25
            pygame.draw.line(self.screen, self.COLOR_GUN, (int(self.GUN_POS[0]), int(self.GUN_POS[1])),
                             (int(end_pos[0]), int(end_pos[1])), 8)

        # Draw projectiles
        for p in self.projectiles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 5, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 5, self.COLOR_PROJECTILE)

        # Draw crosshair
        x, y = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = 12
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x - size, y), (x + size, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x, y - size), (x, y + size), 2)

    def _render_ui(self):
        # Ammo display
        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 10))

        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            if all(t['hit'] for t in self.targets):
                msg = "YOU WIN!"
                color = (0, 150, 0)
            elif self.ammo <= 0 and not self.projectiles:
                msg = "OUT OF AMMO"
                color = (150, 0, 0)
            else:
                msg = "TIME UP"
                color = (150, 0, 0)

            end_text = self.font_msg.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_hit": self.score,
            "targets_remaining": self.NUM_TARGETS - self.score
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # For this to work, you must comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # or set it to a valid one like "x11" or "windows"
    # and ensure you have a display.
    
    # Re-enable display for direct play
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    truncated = False
    clock = pygame.time.Clock()

    print("\n" + "=" * 30)
    print(env.game_description)
    print(env.user_guide)
    print("=" * 30 + "\n")

    while not terminated and not truncated:
        movement = 0  # No-op
        space = 0  # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0]  # shift is not used

        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

        clock.tick(30)  # Limit to 30 FPS for human playability

    env.close()