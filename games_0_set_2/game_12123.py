import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:20:53.544870
# Source Brief: brief_02123.md
# Brief Index: 2123
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bouncing ball.
    The goal is to collect colored orbs for points while avoiding crashing
    into obstacle walls, all within a 60-second time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to collect valuable orbs for points. "
        "Avoid crashing into the walls that appear as time progresses."
    )
    user_guide = "Use the arrow keys (↑↓←→) to apply force and guide the ball."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_GREEN_ORB = (50, 255, 100)
        self.COLOR_RED_ORB = (255, 50, 100)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_TRAIL_START = (100, 150, 255)

        # Player properties
        self.PLAYER_RADIUS = 10
        self.PLAYER_ACCEL = 0.3
        self.PLAYER_FRICTION = 0.99
        self.PLAYER_MAX_SPEED = 8.0

        # Orb properties
        self.ORB_RADIUS = 6
        self.NUM_GREEN_ORBS = 5
        self.NUM_RED_ORBS = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.trail = None
        self.orbs = None
        self.walls = None
        self.steps = None
        self.score = None
        self.current_level_phase = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.current_level_phase = 0
        
        self._respawn_player()
        self._setup_level(1)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        reward = 0.01  # Small reward for surviving
        
        self._handle_input(movement)
        self._update_player_state()
        
        # Collision detection and response
        crash_penalty, orb_reward = self._handle_collisions()
        reward += crash_penalty + orb_reward
        
        self.steps += 1
        
        # Level progression
        self._check_level_progression()
        
        # Termination
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.score >= 10:
                reward += 50  # Victory bonus
            else:
                reward -= 50  # Failure penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.player_vel[1] -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel[1] += self.PLAYER_ACCEL
        elif movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

    def _update_player_state(self):
        # Apply friction
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[1] *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = math.hypot(self.player_vel[0], self.player_vel[1])
        if speed > self.PLAYER_MAX_SPEED:
            scale = self.PLAYER_MAX_SPEED / speed
            self.player_vel[0] *= scale
            self.player_vel[1] *= scale
            
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Update trail
        self.trail.append(list(self.player_pos))
        if len(self.trail) > 20:
            self.trail.pop(0)

    def _handle_collisions(self):
        crash_penalty = 0
        orb_reward = 0
        
        # Outer boundary bounce
        if self.player_pos[0] - self.PLAYER_RADIUS < 0:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -1
        elif self.player_pos[0] + self.PLAYER_RADIUS > self.WIDTH:
            self.player_pos[0] = self.WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -1
            
        if self.player_pos[1] - self.PLAYER_RADIUS < 0:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -1
        elif self.player_pos[1] + self.PLAYER_RADIUS > self.HEIGHT:
            self.player_pos[1] = self.HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -1

        # Internal wall crash
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, 
                                  self.player_pos[1] - self.PLAYER_RADIUS, 
                                  self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        for wall in self.walls:
            if wall.colliderect(player_rect):
                # sfx: crash_sound
                crash_penalty = -2
                self.score -= 2
                self._respawn_player()
                break # only one crash per frame

        # Orb collection
        collected_indices = []
        for i, orb in enumerate(self.orbs):
            dist = math.hypot(self.player_pos[0] - orb['pos'][0], self.player_pos[1] - orb['pos'][1])
            if dist < self.PLAYER_RADIUS + self.ORB_RADIUS:
                collected_indices.append(i)
                if orb['type'] == 'green':
                    # sfx: collect_green_orb
                    self.score += 1
                    orb_reward += 1
                    speed = math.hypot(self.player_vel[0], self.player_vel[1])
                    if speed > 0.1:
                        self.player_vel[0] *= 0.9
                        self.player_vel[1] *= 0.9
                elif orb['type'] == 'red':
                    # sfx: collect_red_orb
                    self.score += 3
                    orb_reward += 3
                    self.player_vel[0] *= 1.2
                    self.player_vel[1] *= 1.2
        
        # Remove collected orbs and spawn new ones
        for i in sorted(collected_indices, reverse=True):
            orb_type = self.orbs.pop(i)['type']
            self._spawn_orb(orb_type, 1)

        return crash_penalty, orb_reward

    def _respawn_player(self):
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.player_vel = [math.cos(angle) * 2, math.sin(angle) * 2]
        self.trail = []

    def _check_level_progression(self):
        new_phase = 0
        if self.steps > self.MAX_STEPS * 2 / 3:
            new_phase = 3
        elif self.steps > self.MAX_STEPS * 1 / 3:
            new_phase = 2
        else:
            new_phase = 1
            
        if new_phase != self.current_level_phase:
            self.current_level_phase = new_phase
            self._setup_level(new_phase)

    def _setup_level(self, level_phase):
        self.walls = []
        if level_phase == 2:
            self.walls.append(pygame.Rect(self.WIDTH/2 - 100, self.HEIGHT/2 - 10, 200, 20))
        elif level_phase == 3:
            self.walls.append(pygame.Rect(self.WIDTH/4, self.HEIGHT/4, 20, self.HEIGHT/2))
            self.walls.append(pygame.Rect(self.WIDTH*3/4 - 20, self.HEIGHT/4, 20, self.HEIGHT/2))

        self.orbs = []
        self._spawn_orb('green', self.NUM_GREEN_ORBS)
        self._spawn_orb('red', self.NUM_RED_ORBS)

    def _spawn_orb(self, orb_type, count):
        for _ in range(count):
            while True:
                pos = [
                    self.np_random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS),
                    self.np_random.uniform(self.ORB_RADIUS, self.HEIGHT - self.ORB_RADIUS)
                ]
                if self._is_valid_spawn(pos, self.ORB_RADIUS):
                    self.orbs.append({'pos': pos, 'type': orb_type})
                    break

    def _is_valid_spawn(self, pos, radius):
        # Check distance from walls
        for wall in self.walls:
            if wall.collidepoint(pos):
                return False
        # Check distance from other orbs
        for orb in self.orbs:
            if math.hypot(pos[0] - orb['pos'][0], pos[1] - orb['pos'][1]) < radius * 3:
                return False
        # Check distance from center (to avoid spawning on player respawn)
        if math.hypot(pos[0] - self.WIDTH/2, pos[1] - self.HEIGHT/2) < 100:
            return False
        return True

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
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        self._render_walls()
        self._render_orbs()
        self._render_trail()
        self._render_player()

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_orbs(self):
        for orb in self.orbs:
            color = self.COLOR_GREEN_ORB if orb['type'] == 'green' else self.COLOR_RED_ORB
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)

    def _render_trail(self):
        if not self.trail:
            return
        for i, pos in enumerate(self.trail):
            if i < 1: continue
            alpha = int(255 * (i / len(self.trail)))
            radius = int(self.PLAYER_RADIUS * 0.7 * (i / len(self.trail)))
            if radius < 1: continue
            
            # Interpolate color from trail start to player color
            trail_color = [
                int(self.COLOR_TRAIL_START[j] + (self.COLOR_PLAYER[j] - self.COLOR_TRAIL_START[j]) * (i / len(self.trail)))
                for j in range(3)
            ]
            
            # Using a separate surface for transparency
            trail_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(trail_surf, radius, radius, radius, (*trail_color, alpha))
            self.screen.blit(trail_surf, (int(pos[0] - radius), int(pos[1] - radius)))

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"Time: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        if self.current_level_phase > 1:
            phase_text = self.font_small.render(f"Phase {self.current_level_phase}", True, self.COLOR_UI)
            self.screen.blit(phase_text, (self.WIDTH/2 - phase_text.get_width()/2, 10))


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bouncing Ball Environment")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False

        # --- Human Input ---
        movement_action = 0  # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()