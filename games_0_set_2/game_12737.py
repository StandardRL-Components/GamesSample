import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:32:56.489938
# Source Brief: brief_02737.md
# Brief Index: 2737
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a charged particle through a dynamically shifting magnetic maze.

    The player controls a particle's horizontal movement to navigate a maze filled
    with oscillating magnetic fields. The goal is to reach the exit before the
    60-second timer runs out.

    **Visuals:**
    - Player: Bright blue glowing circle.
    - Walls: Static light grey lines.
    - Exit: Glowing green square.
    - Magnetic Fields: Pulsating red (repulsive) or blue (attractive) circles.

    **Gameplay:**
    - Player is influenced by magnetic fields.
    - Passing through 3 fields in quick succession triggers a "chain reaction" for a speed boost.
    - Difficulty increases over time as fields oscillate faster.

    **Rewards:**
    - Proximity to exit: Small continuous reward.
    - Passing a field: +1
    - Chain reaction: +5
    - Reaching exit: +100
    - Timeout: -100
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate a charged particle through a shifting magnetic maze to reach the green exit before time runs out."
    )
    user_guide = (
        "Use the ← and → arrow keys to steer the particle through the maze."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (16, 16, 32)  # Dark blue-grey
    COLOR_WALL = (128, 128, 140)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 128, 255)
    COLOR_EXIT = (0, 255, 128)
    COLOR_EXIT_GLOW = (0, 192, 96)
    COLOR_FIELD_ATTRACT = (64, 128, 255)
    COLOR_FIELD_REPEL = (255, 64, 64)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (32, 32, 48)

    # Physics
    PLAYER_RADIUS = 8
    PLAYER_ACCEL = 0.5
    PLAYER_FRICTION = 0.96
    MAX_VELOCITY = 8
    FIELD_BASE_STRENGTH = 15000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 20)
            
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.maze_walls = []
        self.magnetic_fields = []
        self.exit_rect = None
        self.last_dist_to_exit = 0
        self.chain_combo_count = 0
        self.chain_combo_timer = 0
        self.chain_boost_active = False
        self.chain_boost_timer = 0
        self.particles = []

        self._define_maze()
        
    def _define_maze(self):
        """Defines the static layout of the maze walls and exit."""
        self.maze_walls = [
            # Outer boundaries
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT),
            # Inner walls
            pygame.Rect(100, 10, 10, 250),
            pygame.Rect(200, 150, 10, 240),
            pygame.Rect(300, 10, 10, 250),
            pygame.Rect(400, 150, 10, 240),
            pygame.Rect(500, 10, 10, 250),
        ]
        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT / 2 - 20, 40, 40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT / 2])
        self.player_vel = np.array([0.0, 0.0])

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.chain_combo_count = 0
        self.chain_combo_timer = 0
        self.chain_boost_active = False
        self.chain_boost_timer = 0
        self.particles = []
        
        self.last_dist_to_exit = self._get_dist_to_exit()

        # Initialize magnetic fields
        self.magnetic_fields = []
        field_positions = [
            (150, 100), (150, 300),
            (250, 80), (250, 320),
            (350, 100), (350, 300),
            (450, 80), (450, 320),
            (550, 200)
        ]
        for i, pos in enumerate(field_positions):
            self.magnetic_fields.append({
                "pos": np.array(pos, dtype=float),
                "radius": 40,
                "base_strength": self.FIELD_BASE_STRENGTH * (1.0 + self.np_random.uniform(-0.2, 0.2)),
                "phase_offset": self.np_random.uniform(0, 2 * math.pi),
                "player_passed": False
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Update Game Logic & Physics
        self._handle_input(action)
        self._update_physics()
        self._handle_collisions()
        self._update_game_state()
        
        # 2. Calculate Reward
        reward += self._calculate_reward()

        # 3. Check Termination
        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated:
            if self.steps >= self.MAX_STEPS:
                reward -= 100 # Timeout penalty
                # sfx: game_over_timeout
            else: # Reached exit
                reward += 100
                # sfx: win_level
        
        self.score += reward
        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right

        accel_multiplier = 2.0 if self.chain_boost_active else 1.0
        
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL * accel_multiplier
            # sfx: player_move
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL * accel_multiplier
            # sfx: player_move

    def _update_physics(self):
        # Apply magnetic forces
        difficulty_multiplier = 1.0 + (self.steps / (10 * self.FPS)) * 0.1
        
        for field in self.magnetic_fields:
            time_factor = self.steps * 0.05 * difficulty_multiplier + field["phase_offset"]
            polarity = math.sin(time_factor)
            
            vec_to_player = self.player_pos - field["pos"]
            dist_sq = np.dot(vec_to_player, vec_to_player)
            
            if dist_sq < 1: dist_sq = 1 # Avoid division by zero
            
            force_magnitude = polarity * field["base_strength"] / dist_sq
            force_vec = (vec_to_player / math.sqrt(dist_sq)) * force_magnitude
            
            self.player_vel += force_vec / self.FPS

        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION

        # Clamp velocity
        vel_mag = np.linalg.norm(self.player_vel)
        if vel_mag > self.MAX_VELOCITY:
            self.player_vel = self.player_vel / vel_mag * self.MAX_VELOCITY

        # Update position
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_RADIUS,
            self.player_pos[1] - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )

        for wall in self.maze_walls:
            if player_rect.colliderect(wall):
                # sfx: wall_bounce
                # Horizontal collision
                if player_rect.right > wall.left and player_rect.left < wall.left:
                    player_rect.right = wall.left
                    self.player_vel[0] *= -0.5
                elif player_rect.left < wall.right and player_rect.right > wall.right:
                    player_rect.left = wall.right
                    self.player_vel[0] *= -0.5
                
                # Vertical collision
                if player_rect.bottom > wall.top and player_rect.top < wall.top:
                    player_rect.bottom = wall.top
                    self.player_vel[1] *= -0.5
                elif player_rect.top < wall.bottom and player_rect.bottom > wall.bottom:
                    player_rect.top = wall.bottom
                    self.player_vel[1] *= -0.5

                self.player_pos = np.array(player_rect.center, dtype=float)

    def _update_game_state(self):
        # Update chain combo timer
        if self.chain_combo_timer > 0:
            self.chain_combo_timer -= 1
        else:
            self.chain_combo_count = 0

        # Update chain boost
        if self.chain_boost_active:
            self.chain_boost_timer -= 1
            if self.chain_boost_timer <= 0:
                self.chain_boost_active = False
            # Create trail particles
            if self.steps % 2 == 0:
                self._create_particle(self.player_pos, self.COLOR_PLAYER, count=1, speed=0.5)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _get_dist_to_exit(self):
        return np.linalg.norm(self.player_pos - np.array(self.exit_rect.center))

    def _calculate_reward(self):
        reward = 0
        
        # 1. Proximity reward
        current_dist = self._get_dist_to_exit()
        reward += (self.last_dist_to_exit - current_dist) * 0.1
        self.last_dist_to_exit = current_dist
        
        # 2. Magnetic field pass-through reward
        for field in self.magnetic_fields:
            dist_to_field = np.linalg.norm(self.player_pos - field["pos"])
            if dist_to_field < field["radius"]:
                if not field["player_passed"]:
                    field["player_passed"] = True
                    reward += 1.0
                    # sfx: field_enter
                    # Chain combo logic
                    self.chain_combo_count += 1
                    self.chain_combo_timer = 1.5 * self.FPS # 1.5 seconds to hit next field
                    if self.chain_combo_count >= 3:
                        reward += 5.0
                        self.chain_boost_active = True
                        self.chain_boost_timer = 3 * self.FPS # 3 second boost
                        self.chain_combo_count = 0
                        # sfx: chain_reaction
            else:
                field["player_passed"] = False # Reset when player leaves the field
        
        return reward

    def _check_termination(self):
        player_rect = pygame.Rect(0,0, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))

        if player_rect.colliderect(self.exit_rect):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

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
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "player_pos": self.player_pos.tolist(),
        }

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw magnetic fields
        difficulty_multiplier = 1.0 + (self.steps / (10 * self.FPS)) * 0.1
        for field in self.magnetic_fields:
            time_factor = self.steps * 0.05 * difficulty_multiplier + field["phase_offset"]
            polarity = math.sin(time_factor)
            strength = abs(polarity)
            color = self.COLOR_FIELD_ATTRACT if polarity > 0 else self.COLOR_FIELD_REPEL
            
            # Draw pulsating glow
            for i in range(3, 0, -1):
                radius = field["radius"] * (1 + strength * 0.2 * i / 3)
                alpha = 30 * strength * (4 - i) / 3
                pygame.gfxdraw.filled_circle(self.screen, int(field["pos"][0]), int(field["pos"][1]), int(radius), (*color, int(alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(field["pos"][0]), int(field["pos"][1]), int(field["radius"] * 0.3), color)

        # Draw walls
        for wall in self.maze_walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            
        # Draw exit
        glow_size = int(self.exit_rect.width * 1.5)
        pygame.gfxdraw.box(self.screen, self.exit_rect.inflate(glow_size, glow_size), (*self.COLOR_EXIT_GLOW, 50))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Draw player
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 60))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps) / self.FPS
        minutes = int(time_left) // 60
        seconds = int(time_left) % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"
        
        text_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 18, 12))
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_small.render(score_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (22, self.SCREEN_HEIGHT - text_surf.get_height() - 8))
        self.screen.blit(text_surf, (20, self.SCREEN_HEIGHT - text_surf.get_height() - 10))

    def _create_particle(self, pos, color, count=5, speed=2.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'size': self.np_random.integers(2, 5),
                'life': self.np_random.integers(15, 31),
                'max_life': 30
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Maze")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Rendering ---
        # The observation is already the rendered screen, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()