import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ and ↓ to move between lanes. Dodge obstacles and collect coins!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Dodge obstacles and collect coins in a race to the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 60, bold=True)

        # Game constants
        self.MAX_STEPS = 5000
        self.FINISH_LINE_DISTANCE = 15000
        self.INITIAL_LIVES = 3
        self.BASE_SPEED = 2.5
        self.SPEED_INCREASE_INTERVAL = 500
        self.SPEED_INCREASE_AMOUNT = 0.05

        # Colors
        self.COLOR_BG_SKY = (40, 50, 80)
        self.COLOR_ROAD = (80, 80, 90)
        self.COLOR_LANE_MARKER = (200, 200, 210)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 64)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (255, 80, 80, 64)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_COIN_GLOW = (255, 220, 0, 64)
        self.COLOR_UI_TEXT = (255, 255, 255)

        # Lane setup
        self.NUM_LANES = 3
        self.LANE_HEIGHT = 60
        road_top = (self.SCREEN_HEIGHT - self.NUM_LANES * self.LANE_HEIGHT) / 2 + 20
        self.LANE_CENTERS = [road_top + self.LANE_HEIGHT * (i + 0.5) for i in range(self.NUM_LANES)]

        # Initialize state variables
        self.reset()

        # self.validate_implementation() # This is for internal testing and should not be in the final class

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.distance_traveled = 0.0
        self.current_scroll_speed = self.BASE_SPEED

        # Player state
        self.player_lane = 1
        self.player_y = self.LANE_CENTERS[self.player_lane]
        self.player_target_y = self.player_y
        self.player_rect = pygame.Rect(0, 0, 40, 20)
        self.player_rect.center = (self.SCREEN_WIDTH * 0.2, self.player_y)
        self.lane_change_cooldown = 0

        # Entity lists
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.background_elements = self._generate_background()
        self.road_markers = self._generate_road_markers()

        # Spawning logic
        self.next_obstacle_spawn_dist = 200
        self.next_coin_spawn_dist = 100

        # Reward tracking for the current step
        self.just_collided = False
        self.coins_collected_this_step = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, we still need to return a valid observation
            # but we don't process any logic. We just tick the clock.
            self.clock.tick(30)
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        self._handle_input(movement)
        self._update_game_state()

        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.distance_traveled >= self.FINISH_LINE_DISTANCE and self.lives > 0:
                reward += 100  # Goal-oriented reward for finishing

        self.steps += 1
        self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        self.lane_change_cooldown = max(0, self.lane_change_cooldown - 1)
        if self.lane_change_cooldown == 0:
            new_lane = self.player_lane
            if movement == 1:  # Up
                new_lane = max(0, self.player_lane - 1)
            elif movement == 2:  # Down
                new_lane = min(self.NUM_LANES - 1, self.player_lane + 1)

            if new_lane != self.player_lane:
                self.player_lane = new_lane
                self.player_target_y = self.LANE_CENTERS[self.player_lane]
                self.lane_change_cooldown = 5  # Cooldown of 5 frames

    def _update_game_state(self):
        # Reset per-step event trackers
        self.just_collided = False
        self.coins_collected_this_step = 0

        # Update speed based on steps
        self.current_scroll_speed = self.BASE_SPEED + (self.steps // self.SPEED_INCREASE_INTERVAL) * self.SPEED_INCREASE_AMOUNT
        self.distance_traveled += self.current_scroll_speed

        # Update player position (interpolation for smooth movement)
        self.player_y += (self.player_target_y - self.player_y) * 0.4
        self.player_rect.centery = int(self.player_y)

        # Update and spawn entities
        self._update_entities()
        self._update_particles()
        self._spawn_entities()

        # Collision detection
        self._handle_collisions()

    def _update_entities(self):
        # Move obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].x -= self.current_scroll_speed
        self.obstacles = [o for o in self.obstacles if o['rect'].right > 0]

        # Move coins
        for coin in self.coins:
            coin['rect'].x -= self.current_scroll_speed
            coin['angle'] = (coin['angle'] + 5) % 360
        self.coins = [c for c in self.coins if c['rect'].right > 0]

        # Move background elements
        for layer in self.background_elements:
            for element in layer['elements']:
                element['pos'][0] -= self.current_scroll_speed * layer['speed_mod']
                if element['pos'][0] + element['size'][0] < 0:
                    element['pos'][0] = self.SCREEN_WIDTH + random.randint(50, 200)

        # Move road markers
        for marker in self.road_markers:
            marker.x -= self.current_scroll_speed
            if marker.right < 0:
                marker.x = self.SCREEN_WIDTH + 20

    def _update_particles(self):
        # Move and age particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        # Remove dead particles
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_entities(self):
        # Spawn obstacles
        if self.distance_traveled > self.next_obstacle_spawn_dist:
            lane = random.randint(0, self.NUM_LANES - 1)
            rect = pygame.Rect(self.SCREEN_WIDTH, self.LANE_CENTERS[lane] - 15, 50, 30)
            self.obstacles.append({'rect': rect})
            self.next_obstacle_spawn_dist += random.uniform(300, 500)

        # Spawn coins
        if self.distance_traveled > self.next_coin_spawn_dist:
            lane = random.randint(0, self.NUM_LANES - 1)
            # Avoid spawning coins right on top of new obstacles
            can_spawn = True
            for obs in self.obstacles:
                if obs['rect'].left > self.SCREEN_WIDTH - 100 and self.LANE_CENTERS[lane] == obs['rect'].centery:
                    can_spawn = False
                    break
            if can_spawn:
                rect = pygame.Rect(self.SCREEN_WIDTH, self.LANE_CENTERS[lane] - 10, 20, 20)
                self.coins.append({'rect': rect, 'angle': 0})
            self.next_coin_spawn_dist += random.uniform(80, 150)

    def _handle_collisions(self):
        # Obstacle collisions
        collided_obstacle = None
        for obstacle in self.obstacles:
            if self.player_rect.colliderect(obstacle['rect']):
                collided_obstacle = obstacle
                break
        if collided_obstacle:
            self.lives -= 1
            self.just_collided = True
            self.obstacles.remove(collided_obstacle)
            self._create_particles(self.player_rect.center, self.COLOR_OBSTACLE, 30)
            # sfx: explosion

        # Coin collections
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin['rect']):
                collected_coins.append(coin)

        for coin in collected_coins:
            self.score += 1
            self.coins_collected_this_step += 1
            self.coins.remove(coin)
            self._create_particles(coin['rect'].center, self.COLOR_COIN, 15)
            # sfx: coin collect

    def _calculate_reward(self):
        reward = 0.01  # Small reward for surviving a step
        if self.just_collided:
            reward -= 10
        reward += self.coins_collected_this_step * 1
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.distance_traveled >= self.FINISH_LINE_DISTANCE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_SKY)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_road()
        for coin in self.coins:
            self._render_coin(coin)
        for obstacle in self.obstacles:
            self._render_obstacle(obstacle)
        self._render_player()
        self._render_particles()
        self._render_finish_line()

    def _render_background(self):
        for layer in self.background_elements:
            color = layer['color']
            for element in layer['elements']:
                x, y = element['pos']
                w, h = element['size']
                pygame.draw.rect(self.screen, color, (int(x), int(y), int(w), int(h)))

    def _render_road(self):
        road_total_height = self.NUM_LANES * self.LANE_HEIGHT
        road_y_start = (self.SCREEN_HEIGHT - road_total_height) / 2 + 20
        pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, road_y_start, self.SCREEN_WIDTH, road_total_height))

        for marker in self.road_markers:
            pygame.draw.rect(self.screen, self.COLOR_LANE_MARKER, marker)

    def _render_coin(self, coin):
        pulse = 1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.01)
        size = int(coin['rect'].width * pulse)
        glow_size = int(size * 2)

        # Draw glow
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_size // 2, glow_size // 2, glow_size // 2, self.COLOR_COIN_GLOW)
        self.screen.blit(glow_surf, (coin['rect'].centerx - glow_size // 2, coin['rect'].centery - glow_size // 2))

        # Draw coin
        pygame.gfxdraw.filled_circle(self.screen, coin['rect'].centerx, coin['rect'].centery, size // 2, self.COLOR_COIN)
        pygame.gfxdraw.aacircle(self.screen, coin['rect'].centerx, coin['rect'].centery, size // 2, self.COLOR_UI_TEXT)

    def _render_obstacle(self, obstacle):
        rect = obstacle['rect']
        glow_size = 10
        glow_rect = rect.inflate(glow_size, glow_size)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW, (0, 0, glow_rect.width, glow_rect.height), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=4)

    def _render_player(self):
        rect = self.player_rect
        glow_size = 15
        glow_rect = rect.inflate(glow_size, glow_size)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_rect.width, glow_rect.height), border_radius=12)
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=6)
        # Add a "cockpit"
        cockpit_rect = pygame.Rect(0, 0, 15, 12)
        cockpit_rect.center = rect.center
        cockpit_rect.x += 5
        pygame.draw.rect(self.screen, self.COLOR_BG_SKY, cockpit_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'][:3] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (p['pos'][0] - size, p['pos'][1] - size))

    def _render_finish_line(self):
        finish_x = self.SCREEN_WIDTH + (self.FINISH_LINE_DISTANCE - self.distance_traveled)
        if finish_x < self.SCREEN_WIDTH + 50:
            tile_size = 20
            road_total_height = self.NUM_LANES * self.LANE_HEIGHT
            road_y_start = (self.SCREEN_HEIGHT - road_total_height) / 2 + 20

            for i in range(int(road_total_height / tile_size) + 1):
                color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (finish_x, road_y_start + i * tile_size, tile_size, tile_size))
                pygame.draw.rect(self.screen, color, (finish_x + tile_size, road_y_start + i * tile_size, tile_size, tile_size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 100 + i * 30, 12, 20, 10)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, life_rect, border_radius=3)

        # Speed
        speed_text = self.font_ui.render(f"SPEED: {self.current_scroll_speed:.1f}", True, self.COLOR_UI_TEXT)
        speed_rect = speed_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(speed_text, speed_rect)

        # Progress bar
        progress = self.distance_traveled / self.FINISH_LINE_DISTANCE
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.SCREEN_HEIGHT - 40, bar_width, 5))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.SCREEN_HEIGHT - 40, bar_width * progress, 5))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.lives <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"

            end_text = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "distance": self.distance_traveled,
        }

    def _generate_background(self):
        layers = []
        # Far layer
        far_elements = []
        for _ in range(10):
            far_elements.append({
                'pos': [random.randint(0, self.SCREEN_WIDTH), random.randint(20, 100)],
                'size': [random.randint(50, 150), random.randint(40, 80)]
            })
        layers.append({'elements': far_elements, 'speed_mod': 0.1, 'color': (50, 60, 95)})

        # Near layer
        near_elements = []
        for _ in range(8):
            near_elements.append({
                'pos': [random.randint(0, self.SCREEN_WIDTH), random.randint(80, 150)],
                'size': [random.randint(80, 200), random.randint(60, 100)]
            })
        layers.append({'elements': near_elements, 'speed_mod': 0.2, 'color': (60, 70, 105)})
        return layers

    def _generate_road_markers(self):
        markers = []
        road_total_height = self.NUM_LANES * self.LANE_HEIGHT
        road_y_start = (self.SCREEN_HEIGHT - road_total_height) / 2 + 20
        for i in range(1, self.NUM_LANES):
            y = road_y_start + i * self.LANE_HEIGHT - 2
            for j in range(self.SCREEN_WIDTH // 80 + 2):
                markers.append(pygame.Rect(j * 80, y, 40, 4))
        return markers

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'max_life': 30,
                'color': color,
                'size': random.randint(3, 8)
            })

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a pure headless environment
    try:
        env = GameEnv()
        obs, info = env.reset()
        done = False

        # Create a window to display the game
        pygame.display.set_caption("Arcade Racer")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

        total_reward = 0

        action = env.action_space.sample()
        action.fill(0)  # Start with no-op

        clock = pygame.time.Clock()
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            action.fill(0)  # Reset action
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
            # ----------------------

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This game is meant to be run in a headless environment. If you are seeing this,")
        print("it's likely because a display is not available. The code is still valid for training.")