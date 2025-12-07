import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set SDL to dummy mode BEFORE pygame.init() to run headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to steer your boat. Avoid the creatures in the crimson water."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Row your boat down Crimson Creek, dodging lurking horrors to reach the safety of the bridge."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.BRIDGE_DISTANCE = 5000
        self.MAX_STEPS = 1500

        # Colors
        self.COLOR_WATER = (40, 0, 5)
        self.COLOR_BANK = (20, 5, 5)
        self.COLOR_TREES = (30, 10, 10)
        self.COLOR_PLAYER = (210, 210, 220)
        self.COLOR_CREATURE_EYE = (255, 60, 60)
        self.COLOR_CREATURE_BODY = (80, 20, 20)
        self.COLOR_DANGER = (180, 0, 0, 70)  # RGBA for transparency
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_RIPPLE = (60, 10, 15)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        # FIX: Initialize a display mode, even in dummy mode, to allow surface conversions.
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 64)
        except pygame.error: # Fallback if font system fails in some headless environments
            self.font_ui = pygame.font.SysFont("sans", 28)
            self.font_game_over = pygame.font.SysFont("sans", 64)


        # Game parameters
        self.PLAYER_SPEED = 3.0
        self.PLAYER_X_POS = self.WIDTH // 4
        self.BANK_HEIGHT = 40
        self.SCROLL_SPEED = 2.5
        self.DANGER_RADIUS = 70
        self.CREATURE_SPAWN_CHANCE = 0.02
        self.MAX_CREATURES = 10

        # Initialize state variables
        self.player_pos = None
        self.creatures = None
        self.trees = None
        self.ripples = None
        self.world_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.base_creature_speed_multiplier = None

        # self.reset() is called here to ensure initial state is ready
        # This is fine as long as all required attributes are set before reset() is called.
        # No need to call validate_implementation here as it's for dev verification.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.HEIGHT / 2)
        self.creatures = []
        self.ripples = []
        self.world_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_creature_speed_multiplier = 1.0

        # Generate background trees for parallax effect
        self.trees = []
        for _ in range(150):
            self.trees.append({
                'x': self.np_random.integers(0, self.BRIDGE_DISTANCE + self.WIDTH),
                'y': self.np_random.choice([
                    self.np_random.integers(0, self.BANK_HEIGHT - 10),
                    self.np_random.integers(self.HEIGHT - self.BANK_HEIGHT + 10, self.HEIGHT)
                ]),
                'height': self.np_random.integers(15, 40),
                'width': self.np_random.integers(3, 8)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED

        # Clamp player position to stay within the creek
        self.player_pos.y = np.clip(
            self.player_pos.y,
            self.BANK_HEIGHT + 10,
            self.HEIGHT - self.BANK_HEIGHT - 10
        )

        # --- Game Logic Update ---
        self.steps += 1
        previous_world_x = self.world_x
        self.world_x += self.SCROLL_SPEED

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_creature_speed_multiplier += 0.05

        self._update_ripples(action)
        self._update_creatures()

        # --- Collision and Termination Check ---
        reward = self._calculate_reward(previous_world_x)
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.win:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Collision penalty

        self.score += reward

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ripples(self, action):
        # Add new ripples if moving
        if action[0] in [1, 2] and self.steps % 5 == 0:
            ripple_y = self.player_pos.y + self.np_random.uniform(-5, 5)
            self.ripples.append(
                {'pos': pygame.Vector2(self.player_pos.x - 10, ripple_y), 'radius': 1, 'max_radius': 15, 'life': 1.0})

        # Update existing ripples
        self.ripples = [r for r in self.ripples if r['life'] > 0]
        for r in self.ripples:
            r['radius'] += 0.3
            r['life'] -= 0.02

    def _update_creatures(self):
        # Spawn new creatures
        if len(self.creatures) < self.MAX_CREATURES and self.np_random.random() < self.CREATURE_SPAWN_CHANCE:
            y_center = self.np_random.integers(self.BANK_HEIGHT + 30, self.HEIGHT - self.BANK_HEIGHT - 30)
            self.creatures.append({
                'world_x': self.world_x + self.WIDTH + 50,
                'y_center': y_center,
                'amplitude': self.np_random.integers(10, 80),
                'frequency': self.np_random.uniform(0.002, 0.008),
                'phase': self.np_random.uniform(0, 2 * math.pi),
                'speed_multiplier': self.np_random.uniform(0.8, 1.2) * self.base_creature_speed_multiplier,
                'anim_phase': self.np_random.uniform(0, 2 * math.pi)
            })

        # Update and remove off-screen creatures
        new_creatures = []
        for c in self.creatures:
            c['world_x'] -= self.SCROLL_SPEED * c['speed_multiplier']
            if c['world_x'] > self.world_x - 50:
                new_creatures.append(c)
        self.creatures = new_creatures

    def _calculate_reward(self, previous_world_x):
        reward = 0
        # Survival reward
        reward += 0.01

        # Progress reward: +1 per pixel closer to the bridge
        reward += (self.world_x - previous_world_x)

        # Danger zone penalty
        in_danger = False
        for c in self.creatures:
            screen_x = c['world_x'] - self.world_x
            y = c['y_center'] + c['amplitude'] * math.sin(c['world_x'] * c['frequency'] + c['phase'])
            dist = self.player_pos.distance_to((screen_x, y))
            if dist < self.DANGER_RADIUS:
                in_danger = True
                break

        if in_danger:
            reward -= 0.2

        return reward

    def _check_termination(self):
        # Reached max steps is truncation, not termination
        if self.steps >= self.MAX_STEPS:
            return False

        # Reached bridge
        if self.world_x + self.player_pos.x >= self.BRIDGE_DISTANCE:
            self.win = True
            return True

        # Collision with creature
        player_rect = pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 4, 16, 8)
        for c in self.creatures:
            screen_x = c['world_x'] - self.world_x
            y = c['y_center'] + c['amplitude'] * math.sin(c['world_x'] * c['frequency'] + c['phase'])
            # Creatures are rendered as two "eyes", check collision with both
            creature_rect1 = pygame.Rect(screen_x - 8, y - 3, 6, 6)
            creature_rect2 = pygame.Rect(screen_x + 2, y - 3, 6, 6)
            if player_rect.colliderect(creature_rect1) or player_rect.colliderect(creature_rect2):
                return True

        return False

    def _get_observation(self):
        # Clear screen with water color
        self.screen.fill(self.COLOR_WATER)

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        # Pygame array is (width, height, channels), we need (height, width, channels)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw banks
        pygame.draw.rect(self.screen, self.COLOR_BANK, (0, 0, self.WIDTH, self.BANK_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_BANK, (0, self.HEIGHT - self.BANK_HEIGHT, self.WIDTH, self.BANK_HEIGHT))

        # Draw trees
        for tree in self.trees:
            screen_x = int(tree['x'] - self.world_x)
            if 0 <= screen_x <= self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_TREES,
                                 (screen_x, tree['y'] - tree['height'] // 2, tree['width'], tree['height']))

        # Draw ripples
        for r in self.ripples:
            screen_x = int(r['pos'].x)
            alpha = int(max(0, 255 * r['life'] * (1 - r['radius'] / r['max_radius'])))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, screen_x, int(r['pos'].y), int(r['radius']),
                                        (*self.COLOR_RIPPLE, alpha))

        # Draw danger zones and creatures
        danger_surface = self.screen.convert_alpha()
        danger_surface.fill((0, 0, 0, 0))
        for c in self.creatures:
            screen_x = c['world_x'] - self.world_x
            if -self.DANGER_RADIUS < screen_x < self.WIDTH + self.DANGER_RADIUS:
                y = c['y_center'] + c['amplitude'] * math.sin(c['world_x'] * c['frequency'] + c['phase'])
                # Draw danger zone
                pygame.draw.circle(danger_surface, self.COLOR_DANGER, (int(screen_x), int(y)), self.DANGER_RADIUS)
                # Draw creature
                anim_offset = 3 * math.sin(pygame.time.get_ticks() * 0.01 + c['anim_phase'])
                pygame.draw.ellipse(self.screen, self.COLOR_CREATURE_BODY, (screen_x - 10, y - 5 + anim_offset, 20, 10))
                pygame.draw.circle(self.screen, self.COLOR_CREATURE_EYE, (int(screen_x - 4), int(y + anim_offset)), 3)
                pygame.draw.circle(self.screen, self.COLOR_CREATURE_EYE, (int(screen_x + 4), int(y + anim_offset)), 3)
        self.screen.blit(danger_surface, (0, 0))

        # Draw bridge
        bridge_screen_x = self.BRIDGE_DISTANCE - self.world_x
        if bridge_screen_x < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_TREES, (bridge_screen_x, 0, 20, self.HEIGHT))
            for i in range(10):
                pygame.draw.line(self.screen, self.COLOR_BANK,
                                 (bridge_screen_x + 10, i * 45),
                                 (bridge_screen_x + 10, i * 45 + 30), 4)

        # Draw player
        player_rect = pygame.Rect(0, 0, 20, 10)
        player_rect.center = self.player_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.circle(self.screen, self.COLOR_WATER, (int(self.player_pos.x + 5), int(self.player_pos.y)), 2)

    def _render_ui(self):
        # Distance display
        dist_to_go = max(0, self.BRIDGE_DISTANCE - self.world_x - self.player_pos.x)
        dist_text = f"Distance: {int(dist_to_go)}"
        dist_surf = self.font_ui.render(dist_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_surf, (10, 10))

        # Score display
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU REACHED THE BRIDGE" if self.win else "THE CREEK CLAIMS YOU"
            color = (180, 255, 180) if self.win else (255, 100, 100)
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_bridge": max(0, self.BRIDGE_DISTANCE - self.world_x - self.player_pos.x),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a visible window for playing
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # The environment's self.screen is now the main display surface
    # We can use it directly for interactive play.
    pygame.display.set_caption("Crimson Creek")
    
    running = True
    total_reward = 0

    # --- Instructions ---
    print("\n" + "=" * 30)
    print("      CRIMSON CREEK")
    print("=" * 30)
    print(env.game_description)
    print(env.user_guide)
    print("=" * 30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        if keys[pygame.K_r]:  # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")
        if keys[pygame.K_ESCAPE]:
            running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30)  # Run at 30 FPS

    env.close()