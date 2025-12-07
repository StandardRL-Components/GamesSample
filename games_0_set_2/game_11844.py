import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a cursor to collect glowing nodes.
    Collecting nodes increases speed and spawns more nodes. The goal is to collect
    the initial set of nodes before speed drops to zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a glowing cursor to collect nodes. Each node collected increases your speed, "
        "but your speed constantly drains."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your cursor."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    INITIAL_NODES = 25
    MAX_NODES = 200

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_NODE_DIM = (0, 100, 255)
    COLOR_NODE_BRIGHT = (255, 255, 0)
    COLOR_NODE_COLLECTED = (50, 50, 60)
    COLOR_TEXT = (200, 200, 220)

    # --- Game Parameters ---
    PLAYER_SIZE = 8
    INITIAL_SPEED = 4.0
    SPEED_GAIN_ON_COLLECT = 1.20  # 20% increase
    SPEED_LOSS_ON_BOUNDARY = 0.90 # 10% decrease
    NODE_MIN_RADIUS = 8
    NODE_MAX_RADIUS = 15
    NODE_PROXIMITY_EFFECT_RANGE = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)

        # Game state variables initialized in reset()
        self.player_pos = None
        self.player_speed = None
        self.nodes = None
        self.particles = None
        self.steps = None
        self.score = None
        self.initial_nodes_collected = None
        self.last_distance_to_nearest = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_speed = self.INITIAL_SPEED

        self.nodes = []
        for i in range(self.INITIAL_NODES):
            self._spawn_node(is_initial=True)

        self.particles = []
        self.steps = 0
        self.score = 0
        self.initial_nodes_collected = 0

        _, self.last_distance_to_nearest = self._get_nearest_node_info()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action  # space and shift are unused

        reward = 0
        terminated = False

        # --- 1. Calculate pre-move state for reward ---
        _, dist_before = self._get_nearest_node_info()

        # --- 2. Update player position ---
        if self.player_speed > 0:
            move_vector = np.array([0, 0], dtype=np.float32)
            if movement == 1:  # Up
                move_vector[1] = -1
            elif movement == 2:  # Down
                move_vector[1] = 1
            elif movement == 3:  # Left
                move_vector[0] = -1
            elif movement == 4:  # Right
                move_vector[0] = 1

            self.player_pos += move_vector * self.player_speed

        # --- 3. Handle collisions and game events ---
        # Boundary collision
        original_pos = self.player_pos.copy()
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)
        if not np.array_equal(original_pos, self.player_pos):
            self.player_speed *= self.SPEED_LOSS_ON_BOUNDARY
            reward -= 1.0

        # Node collection
        for node in self.nodes:
            if not node['collected']:
                dist = np.linalg.norm(self.player_pos - node['pos'])
                if dist < node['radius'] + self.PLAYER_SIZE / 2:
                    node['collected'] = True
                    self.score += 1
                    reward += 1.0
                    self.player_speed *= self.SPEED_GAIN_ON_COLLECT
                    self._create_particles(node['pos'], self.COLOR_NODE_BRIGHT)

                    if node['is_initial']:
                        self.initial_nodes_collected += 1

                    if len(self.nodes) < self.MAX_NODES:
                        self._spawn_node(is_initial=False)
                    break # Collect one node per step

        # --- 4. Update particles ---
        self._update_particles()

        # --- 5. Calculate proximity reward ---
        _, dist_after = self._get_nearest_node_info()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.2
            self.last_distance_to_nearest = dist_after

        # --- 6. Check for termination conditions ---
        self.steps += 1
        win_condition = self.initial_nodes_collected >= self.INITIAL_NODES
        loss_condition = self.player_speed < 0.1
        timeout = self.steps >= self.MAX_STEPS

        if win_condition:
            reward += 100.0
            terminated = True
        elif loss_condition:
            reward -= 100.0
            terminated = True
        elif timeout:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
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
            "player_speed": self.player_speed,
            "initial_nodes_collected": self.initial_nodes_collected
        }

    def _spawn_node(self, is_initial):
        # Ensure new nodes don't spawn too close to the player
        while True:
            pos = self.np_random.uniform(
                low=[self.NODE_MAX_RADIUS, self.NODE_MAX_RADIUS],
                high=[self.SCREEN_WIDTH - self.NODE_MAX_RADIUS, self.SCREEN_HEIGHT - self.NODE_MAX_RADIUS],
                size=(2,)
            ).astype(np.float32)
            if self.player_pos is None or np.linalg.norm(pos - self.player_pos) > 100:
                break

        self.nodes.append({
            'pos': pos,
            'radius': self.np_random.uniform(self.NODE_MIN_RADIUS, self.NODE_MAX_RADIUS),
            'collected': False,
            'is_initial': is_initial
        })

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render nodes
        for node in self.nodes:
            dist_to_player = np.linalg.norm(self.player_pos - node['pos'])

            if node['collected']:
                color = self.COLOR_NODE_COLLECTED
            else:
                # Energize node color based on proximity
                energy = 1.0 - np.clip(dist_to_player / self.NODE_PROXIMITY_EFFECT_RANGE, 0, 1)
                color = self._lerp_color(self.COLOR_NODE_DIM, self.COLOR_NODE_BRIGHT, energy**2)

            self._draw_glow_circle(node['pos'], node['radius'], color)

        # Render player
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        self._draw_glow_circle(self.player_pos, self.PLAYER_SIZE, self.COLOR_PLAYER, intensity=0.3)

    def _render_ui(self):
        # Speed display
        speed_text = f"Speed: {self.player_speed:.1f}"
        speed_surf = self.font.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (10, 10))

        # Score display
        score_text = f"Collected: {self.initial_nodes_collected} / {self.INITIAL_NODES}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

    def _get_nearest_node_info(self):
        uncollected_nodes = [n for n in self.nodes if not n['collected']]
        if not uncollected_nodes:
            return None, None

        distances = [np.linalg.norm(self.player_pos - n['pos']) for n in uncollected_nodes]
        min_idx = np.argmin(distances)

        return uncollected_nodes[min_idx], distances[min_idx]

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'max_lifetime': self.np_random.uniform(15, 30),
                'lifetime': self.np_random.uniform(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.98 # Shrink
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    @staticmethod
    def _lerp_color(c1, c2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def _draw_glow_circle(self, pos, radius, color, intensity=1.0):
        x, y = int(pos[0]), int(pos[1])

        # Draw multiple layers for glow effect
        for i in range(int(radius * 1.5), 0, -2):
            alpha = int(50 * (1 - (i / (radius * 2.5))) * intensity)
            if alpha <= 0: continue

            glow_color = (color[0], color[1], color[2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, x, y, i, glow_color)

        # Draw main circle
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    # This block will not run in a headless environment, but is useful for local testing.
    # To run, you might need to comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    # and ensure you have a display.
    try:
        env = GameEnv()
        obs, info = env.reset()

        pygame.display.set_caption("Node Collector")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        terminated = False
        total_reward = 0

        print("\n--- Manual Control ---")
        print("Use arrow keys to move.")
        print("Press Q or close window to quit.")

        while not terminated:
            movement_action = 0 # no-op

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement_action = 1
            elif keys[pygame.K_DOWN]:
                movement_action = 2
            elif keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
            if keys[pygame.K_q]:
                terminated = True

            action = [movement_action, 0, 0] # Space and Shift are not used

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term or terminated

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit to 30 FPS

        print(f"\nGame Over!")
        print(f"Final Info: {info}")
        print(f"Total Reward: {total_reward:.2f}")

        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This example requires a display. It may not run in a headless environment.")