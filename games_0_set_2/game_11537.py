import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:12:08.780104
# Source Brief: brief_01537.md
# Brief Index: 1537
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a creature navigates a 2D maze with a 3D-like
    rendering style. The goal is to collect all crystals to unlock a sonar ability,
    and then reach the exit before time runs out. The environment is designed for
    visual appeal and satisfying game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a creature through a dark maze, collecting all crystals to unlock a sonar ability "
        "and find the exit before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move and turn. After collecting all crystals, "
        "press space to use sonar and locate the exit."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_FLOOR = (40, 45, 60)
    COLOR_WALL_TOP = (60, 65, 80)
    COLOR_WALL_FRONT = (50, 55, 70)
    COLOR_ROUGH = (30, 35, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_CRYSTAL = (255, 220, 0)
    COLOR_CRYSTAL_GLOW = (255, 220, 0, 60)
    COLOR_EXIT = (255, 255, 255)
    COLOR_EXIT_GLOW = (255, 255, 255, 80)
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_UI_TEXT = (220, 220, 230)

    # Game Parameters
    MAX_STEPS = 90 * FPS  # 90 seconds
    NUM_CRYSTALS = 12
    WALL_THICKNESS = 20
    WALL_RENDER_HEIGHT = 15

    # Player Physics
    PLAYER_RADIUS = 10
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.95
    PLAYER_ROUGH_FRICTION = 0.85
    PLAYER_ROTATION_SPEED = 0.1
    PLAYER_MAX_SPEED = 5.0
    PLAYER_BOUNCE = -0.5

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        self.render_mode = render_mode
        self.sonar_pulses = []
        self.walls = []
        self.rough_patches = []
        self.crystals = []
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.player_angle = 0
        self.exit_pos = [0, 0]
        self.all_crystals_collected = False
        self.last_dist_to_target = float('inf')

        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is for dev, not needed in final code

    def _generate_maze(self):
        """Creates a static, solvable maze layout."""
        self.walls = []
        # Outer boundaries
        self.walls.append(pygame.Rect(0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        self.walls.append(pygame.Rect(0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        self.walls.append(pygame.Rect(0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        self.walls.append(pygame.Rect(self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        # Inner walls
        self.walls.append(pygame.Rect(150, 0, self.WALL_THICKNESS, 200))
        self.walls.append(pygame.Rect(self.SCREEN_WIDTH - 150 - self.WALL_THICKNESS, self.SCREEN_HEIGHT - 200, self.WALL_THICKNESS, 200))
        self.walls.append(pygame.Rect(250, 150, 140, self.WALL_THICKNESS))

        self.rough_patches = []
        self.rough_patches.append(pygame.Rect(200, 250, 150, 100))
        self.rough_patches.append(pygame.Rect(50, 50, 80, 120))
        self.rough_patches.append(pygame.Rect(450, 80, 120, 80))

    def _is_valid_spawn(self, x, y, radius, min_dist=30):
        """Checks if a position is valid for spawning an object."""
        if not (radius < x < self.SCREEN_WIDTH - radius and radius < y < self.SCREEN_HEIGHT - radius):
            return False
        point_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        if any(point_rect.colliderect(wall) for wall in self.walls):
            return False
        if any(math.hypot(x - c['pos'][0], y - c['pos'][1]) < min_dist for c in self.crystals):
            return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.all_crystals_collected = False
        self.sonar_pulses = []

        self._generate_maze()

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_vel = [0, 0]
        self.player_angle = -math.pi / 2  # Start facing up

        self.crystals = []
        while len(self.crystals) < self.NUM_CRYSTALS:
            x = self.np_random.uniform(self.WALL_THICKNESS + 15, self.SCREEN_WIDTH - self.WALL_THICKNESS - 15)
            y = self.np_random.uniform(self.WALL_THICKNESS + 15, self.SCREEN_HEIGHT - self.WALL_THICKNESS - 15)
            if self._is_valid_spawn(x, y, 10, min_dist=40):
                self.crystals.append({'pos': [x, y], 'radius': 7})

        self.exit_pos = [self.SCREEN_WIDTH / 2, 50]

        self.last_dist_to_target = self._get_dist_to_target()

        return self._get_observation(), self._get_info()

    def _get_dist_to_target(self):
        """Calculates distance to the nearest crystal or the exit."""
        if not self.all_crystals_collected:
            if not self.crystals: return 0
            return min(math.hypot(self.player_pos[0] - c['pos'][0], self.player_pos[1] - c['pos'][1]) for c in self.crystals)
        else:
            return math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # --- Update player based on action ---
        force = [0, 0]
        if movement == 1:  # Forward
            force = [math.cos(self.player_angle) * self.PLAYER_ACCEL, math.sin(self.player_angle) * self.PLAYER_ACCEL]
        elif movement == 2:  # Backward
            force = [-math.cos(self.player_angle) * self.PLAYER_ACCEL * 0.7, -math.sin(self.player_angle) * self.PLAYER_ACCEL * 0.7]
        if movement == 3:  # Rotate Left
            self.player_angle -= self.PLAYER_ROTATION_SPEED
        elif movement == 4:  # Rotate Right
            self.player_angle += self.PLAYER_ROTATION_SPEED

        self.player_vel[0] += force[0]
        self.player_vel[1] += force[1]
        
        speed = math.hypot(*self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel[0] = (self.player_vel[0] / speed) * self.PLAYER_MAX_SPEED
            self.player_vel[1] = (self.player_vel[1] / speed) * self.PLAYER_MAX_SPEED

        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        friction = self.PLAYER_ROUGH_FRICTION if any(p.colliderect(player_rect) for p in self.rough_patches) else self.PLAYER_FRICTION
        self.player_vel[0] *= friction
        self.player_vel[1] *= friction

        old_pos = list(self.player_pos)
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # --- Collision Detection ---
        player_rect.center = self.player_pos
        for wall in self.walls:
            if player_rect.colliderect(wall):
                self.player_pos = old_pos
                self.player_vel[0] *= self.PLAYER_BOUNCE
                self.player_vel[1] *= self.PLAYER_BOUNCE
                reward -= 0.01
                # sfx: player_bonk.wav
                break

        # --- Collect Crystals ---
        for crystal in self.crystals[:]:
            dist = math.hypot(self.player_pos[0] - crystal['pos'][0], self.player_pos[1] - crystal['pos'][1])
            if dist < self.PLAYER_RADIUS + crystal['radius']:
                self.crystals.remove(crystal)
                self.score += 1
                reward += 10
                # sfx: crystal_collect.wav
                if not self.crystals:
                    self.all_crystals_collected = True
                    # sfx: all_crystals_collected_chime.wav

        # --- Reward for getting closer ---
        current_dist = self._get_dist_to_target()
        if current_dist < self.last_dist_to_target:
            reward += 0.1 if not self.all_crystals_collected else 0.01
        self.last_dist_to_target = current_dist

        # --- Sonar Action ---
        if space_held and self.all_crystals_collected:
            if not any(p['active'] for p in self.sonar_pulses):
                self.sonar_pulses.append({'pos': list(self.player_pos), 'radius': 0, 'max_radius': 300, 'lifetime': 1.0, 'active': True})
                # sfx: sonar_ping.wav
        
        # --- Update Sonar Pulses ---
        for pulse in self.sonar_pulses[:]:
            pulse['radius'] += pulse['max_radius'] / (self.FPS * pulse['lifetime'])
            if pulse['radius'] >= pulse['max_radius']:
                self.sonar_pulses.remove(pulse)

        # --- Termination Conditions ---
        terminated = False
        truncated = False
        exit_dist = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])
        if self.all_crystals_collected and exit_dist < self.PLAYER_RADIUS + 10:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: victory.wav
        
        if self.steps >= self.MAX_STEPS:
            if not terminated: # Don't penalize if won on last step
                reward -= 100
            terminated = True
            self.game_over = True
            # sfx: timeout_fail.wav

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _render_game(self):
        """Renders all game objects with a pseudo-3D perspective."""
        # Floor and rough patches
        self.screen.fill(self.COLOR_FLOOR)
        for patch in self.rough_patches:
            pygame.draw.rect(self.screen, self.COLOR_ROUGH, patch)

        renderables = []
        for wall in self.walls:
            renderables.append(('wall', wall.y + wall.h, wall))
        for i, crystal in enumerate(self.crystals):
            renderables.append(('crystal', crystal['pos'][1], crystal, i))
        renderables.append(('exit', self.exit_pos[1], self.exit_pos))
        renderables.append(('player', self.player_pos[1], self.player_pos))
        
        renderables.sort(key=lambda x: x[1])

        for r_type, _, *r_data in renderables:
            if r_type == 'wall':
                wall = r_data[0]
                pygame.draw.rect(self.screen, self.COLOR_WALL_FRONT, (wall.x, wall.y, wall.w, wall.h))
                pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (wall.x, wall.y - self.WALL_RENDER_HEIGHT, wall.w, self.WALL_RENDER_HEIGHT), 0, border_radius=2)
            
            elif r_type == 'crystal':
                crystal, i = r_data
                pos = (int(crystal['pos'][0]), int(crystal['pos'][1]))
                pulse_size = 2 * math.sin(self.steps * 0.1 + i)
                radius = int(crystal['radius'] + pulse_size)
                
                shadow_rect = pygame.Rect(pos[0] - radius, pos[1] + radius // 2, radius * 2, radius)
                shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, shadow_rect.width, shadow_rect.height))
                self.screen.blit(shadow_surface, shadow_rect.topleft)

                for j in range(4, 0, -1):
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + j * 2, (*self.COLOR_CRYSTAL_GLOW[:3], self.COLOR_CRYSTAL_GLOW[3] // j))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CRYSTAL)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CRYSTAL)

            elif r_type == 'exit':
                pos = (int(r_data[0][0]), int(r_data[0][1]))
                radius = 15
                if self.all_crystals_collected:
                    glow_radius = radius + 10 + 5 * math.sin(self.steps * 0.2)
                    for j in range(6, 0, -1):
                        alpha = self.COLOR_EXIT_GLOW[3] // j
                        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(glow_radius * (j/6)), (*self.COLOR_EXIT_GLOW[:3], alpha))
                
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_EXIT)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_EXIT)

            elif r_type == 'player':
                pos = (int(r_data[0][0]), int(r_data[0][1]))
                radius = self.PLAYER_RADIUS
                
                shadow_rect = pygame.Rect(pos[0] - radius, pos[1] + radius // 2, radius * 2, radius)
                shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, shadow_rect.width, shadow_rect.height))
                self.screen.blit(shadow_surface, shadow_rect.topleft)

                for j in range(4, 0, -1):
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + j * 3, (*self.COLOR_PLAYER_GLOW[:3], self.COLOR_PLAYER_GLOW[3] // j))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

                end_pos_x = pos[0] + math.cos(self.player_angle) * (radius + 3)
                end_pos_y = pos[1] + math.sin(self.player_angle) * (radius + 3)
                pygame.draw.line(self.screen, self.COLOR_BG, pos, (int(end_pos_x), int(end_pos_y)), 3)

        for pulse in self.sonar_pulses:
            alpha = max(0, 255 * (1 - (pulse['radius'] / pulse['max_radius'])))
            color = (*self.COLOR_PLAYER, int(alpha))
            radius = int(pulse['radius'])
            pygame.gfxdraw.aacircle(self.screen, int(pulse['pos'][0]), int(pulse['pos'][1]), radius, color)
            if radius > 1:
                pygame.gfxdraw.aacircle(self.screen, int(pulse['pos'][0]), int(pulse['pos'][1]), radius - 1, color)

    def _render_ui(self):
        """Renders the UI overlay."""
        crystal_text = self.font_ui.render(f"Crystals: {self.score}/{self.NUM_CRYSTALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_ui.render(f"Time: {max(0, time_left):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            msg, color = ("VICTORY", self.COLOR_CRYSTAL) if self.all_crystals_collected and self.score == self.NUM_CRYSTALS else ("TIME OUT", (255, 100, 100))
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_text = self.font_msg.render(msg, True, (0,0,0))
            self.screen.blit(shadow_text, (msg_rect.x + 2, msg_rect.y + 2))
            self.screen.blit(msg_text, msg_rect)

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
            "crystals_left": len(self.crystals),
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Unset the dummy video driver to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Blind Creature's Maze")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Determine movement action. Note: diagonal movement is not a single action.
        # The agent must learn to alternate left/right while moving forward.
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds before quitting
            break
            
    env.close()