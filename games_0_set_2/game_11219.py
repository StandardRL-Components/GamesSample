import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math

# --- Game Constants ---

# Screen Dimensions
WIDTH, HEIGHT = 640, 400

# Colors (Bio-Electric Theme)
COLOR_BG = (5, 10, 20)
COLOR_PLAYER = (0, 150, 255)
COLOR_SMALL_NEURON = (50, 255, 50)
COLOR_LARGE_NEURON = (255, 50, 50)
COLOR_TEXT = (220, 220, 240)
COLOR_PARTICLE = (200, 220, 255)

# Game Parameters
MAX_STEPS = 3600  # 120 seconds @ 30 FPS
PLAYER_SPEED = 4.0
PLAYER_INITIAL_SIZE = 15.0
SMALL_NEURON_COUNT = 15
SMALL_NEURON_SIZE = 7.0
LARGE_NEURON_INITIAL_SIZE = 25.0
LARGE_NEURON_INITIAL_SPEED = 1.0

# RL Reward Parameters
REWARD_ABSORB = 0.1
REWARD_CHAIN = 1.0
REWARD_WIN = 100.0
REWARD_LOSE = -100.0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a bio-electric world, absorbing small neurons to grow while avoiding contact with larger, hostile ones."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your cell and absorb smaller neurons."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 28)
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        if self.render_mode == "human":
            # This check is to run headlessly if needed
            if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
                # In a dummy environment, we can't create a display surface.
                # We'll just render to self.screen and it will be captured as rgb_array.
                pass
            else:
                pygame.display.set_caption("Neural Nexus")
                self.display_screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = None
        self.player_size = None
        self.firing_speed = None
        self.small_neurons = []
        self.large_neurons = []
        self.particles = []
        self.total_absorbed = 0
        self.chain_absorptions = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player_pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=np.float32)
        self.player_size = PLAYER_INITIAL_SIZE
        self.firing_speed = 1.0

        self.small_neurons = []
        for _ in range(SMALL_NEURON_COUNT):
            self._spawn_small_neuron()

        self.large_neurons = []
        self.particles = []
        self.total_absorbed = 0
        self.chain_absorptions = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement = action[0]
        self._update_player(movement)

        # --- Update Game Entities ---
        self._update_large_neurons()
        self._update_particles()

        # --- Collision Detection and Logic ---
        reward += self._handle_collisions()

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= MAX_STEPS

        if terminated:
            reward = REWARD_LOSE
            self.game_over = True
        elif truncated:
            reward = REWARD_WIN
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_player(self, movement):
        direction = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            direction[1] = -1
        elif movement == 2:  # Down
            direction[1] = 1
        elif movement == 3:  # Left
            direction[0] = -1
        elif movement == 4:  # Right
            direction[0] = 1
        
        self.player_pos += direction * PLAYER_SPEED
        # World wrap-around
        self.player_pos[0] %= WIDTH
        self.player_pos[1] %= HEIGHT

    def _update_large_neurons(self):
        speed_multiplier = 1.0 + 0.05 * (self.steps // 200)
        for neuron in self.large_neurons:
            neuron['pos'] += neuron['vel'] * speed_multiplier
            neuron['pos'][0] %= WIDTH
            neuron['pos'][1] %= HEIGHT
            # Slightly alter velocity for organic movement
            if self.np_random.random() < 0.02:
                angle_change = self.np_random.uniform(-0.5, 0.5)
                cos_a, sin_a = math.cos(angle_change), math.sin(angle_change)
                x, y = neuron['vel']
                neuron['vel'][0] = x * cos_a - y * sin_a
                neuron['vel'][1] = x * sin_a + y * cos_a


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] += 0.2

    def _handle_collisions(self):
        reward = 0.0
        
        # Player vs Small Neurons
        absorbed_indices = []
        for i, neuron in enumerate(self.small_neurons):
            dist = np.linalg.norm(self.player_pos - neuron['pos'])
            if dist < self.player_size + SMALL_NEURON_SIZE:
                absorbed_indices.append(i)
        
        if absorbed_indices:
            # Process absorptions
            for i in sorted(absorbed_indices, reverse=True):
                self._spawn_particles(self.small_neurons[i]['pos'], 10, COLOR_SMALL_NEURON)
                del self.small_neurons[i]
                self._spawn_small_neuron()

                reward += REWARD_ABSORB
                self.player_size = min(PLAYER_INITIAL_SIZE * 3, self.player_size + 0.25)
                self.total_absorbed += 1
                self.chain_absorptions += 1

                if self.chain_absorptions > 0 and self.chain_absorptions % 5 == 0:
                    reward += REWARD_CHAIN
                    self.firing_speed *= 1.2
                
                if self.total_absorbed > 0 and self.total_absorbed % 5 == 0:
                    self._spawn_large_neuron()
        
        # Player vs Large Neurons
        for neuron in self.large_neurons:
            dist = np.linalg.norm(self.player_pos - neuron['pos'])
            if dist < self.player_size + neuron['size']:
                self.game_over = True
                self._spawn_particles(self.player_pos, 50, COLOR_PLAYER)
                break
        
        return reward

    def _check_termination(self):
        # Termination is now only collision with a large neuron
        return self.game_over

    def _spawn_small_neuron(self):
        pos = self.np_random.uniform(low=0, high=[WIDTH, HEIGHT]).astype(np.float32)
        self.small_neurons.append({'pos': pos})

    def _spawn_large_neuron(self):
        # Spawn away from the player
        while True:
            pos = self.np_random.uniform(low=0, high=[WIDTH, HEIGHT]).astype(np.float32)
            if np.linalg.norm(pos - self.player_pos) > self.player_size + LARGE_NEURON_INITIAL_SIZE + 50:
                break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * LARGE_NEURON_INITIAL_SPEED
        self.large_neurons.append({
            'pos': pos, 
            'vel': vel, 
            'size': LARGE_NEURON_INITIAL_SIZE
        })

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(1, 3),
                'color': color
            })

    def _get_observation(self):
        self._render_game_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_absorbed": self.total_absorbed,
            "firing_speed": self.firing_speed,
        }

    def _render_game_surface(self):
        self.screen.fill(COLOR_BG)

        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Render Small Neurons
        for neuron in self.small_neurons:
            self._draw_glowing_circle(
                self.screen, 
                neuron['pos'], 
                SMALL_NEURON_SIZE, 
                COLOR_SMALL_NEURON, 
                pulse_amp=2, 
                pulse_freq=0.5
            )

        # Render Large Neurons
        for neuron in self.large_neurons:
            self._draw_glowing_circle(
                self.screen, 
                neuron['pos'], 
                neuron['size'], 
                COLOR_LARGE_NEURON, 
                pulse_amp=4, 
                pulse_freq=2.0
            )

        # Render Player
        if not self.game_over:
            self._draw_glowing_circle(
                self.screen, 
                self.player_pos, 
                self.player_size, 
                COLOR_PLAYER, 
                pulse_amp=3, 
                pulse_freq=self.firing_speed
            )

        # Render UI
        self._render_ui()

    def _render_ui(self):
        time_left = max(0, (MAX_STEPS - self.steps) // self.metadata['render_fps'])
        timer_text = self.font.render(f"TIME: {time_left}", True, COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))

        absorbed_text = self.font.render(f"ABSORBED: {self.total_absorbed}", True, COLOR_TEXT)
        self.screen.blit(absorbed_text, (10, 40))

        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _draw_glowing_circle(self, surface, pos, radius, color, pulse_amp, pulse_freq):
        pos_int = (int(pos[0]), int(pos[1]))
        
        pulse = pulse_amp * math.sin(self.steps * 0.1 * pulse_freq)
        current_radius = max(1, radius + pulse)

        for i in range(4):
            alpha = 60 - i * 15
            glow_radius = int(current_radius + i * 4)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], glow_radius, glow_color)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], glow_radius, glow_color)
        
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(current_radius), color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(current_radius), color)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # This check is to prevent crashing when running headlessly
        if self.render_mode == "human" and "display_screen" in self.__dict__:
            self._render_game_surface()
            self.display_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0

    action = np.array([0, 0, 0])

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT: action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: action[1] = 0
                if event.key == pygame.K_LSHIFT: action[2] = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            done = True

    env.close()