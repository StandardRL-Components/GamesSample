import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:49:58.465189
# Source Brief: brief_01941.md
# Brief Index: 1941
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Aim your staff and fire splitting projectiles to destroy falling targets before they reach the bottom of the screen."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim the staff and press space to fire projectiles."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_TARGET_COUNT = 15

        # Colors
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_STAFF = (255, 255, 255)
        self.COLOR_PROJECTILE = (0, 255, 255)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_DESTINATION_LINE = (100, 20, 20)
        self.COLOR_GAMEOVER_TEXT = (255, 255, 100)
        self.COLOR_WIN_TEXT = (100, 255, 100)
        
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
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.loss = None
        
        self.staff_pos = None
        self.staff_angle = None
        self.staff_rotation_speed = 2.0

        self.projectiles = None
        self.targets = None
        self.particles = None

        self.launch_cooldown = None
        self.launch_cooldown_max = 10

        self.target_spawn_timer = None
        self.target_base_speed = None
        self.target_current_speed = None
        self.targets_destroyed = None
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.loss = False
        
        self.staff_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        self.staff_angle = -90.0  # Pointing straight up

        self.projectiles = []
        self.targets = []
        self.particles = []

        self.launch_cooldown = 0
        self.targets_destroyed = 0
        
        self.target_base_speed = 1.0
        self.target_current_speed = self.target_base_speed
        self.target_spawn_timer = 30

        # Spawn initial targets
        for _ in range(3):
            self._spawn_target()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3:  # Left
            self.staff_angle -= self.staff_rotation_speed
        elif movement == 4: # Right
            self.staff_angle += self.staff_rotation_speed
        
        # Clamp staff angle to prevent pointing down
        self.staff_angle = max(-170, min(-10, self.staff_angle))

        if space_pressed and self.launch_cooldown <= 0:
            self._launch_projectile()
            self.launch_cooldown = self.launch_cooldown_max
            # sfx: player_shoot.wav

        # --- Game Logic Update ---
        self.launch_cooldown = max(0, self.launch_cooldown - 1)
        self.target_spawn_timer = max(0, self.target_spawn_timer - 1)

        # Update and manage projectiles
        reward += self._update_projectiles()

        # Update and manage targets
        reward += self._update_targets()

        # Update particles
        self._update_particles()
        
        # Spawn new targets
        if self.target_spawn_timer <= 0:
            self._spawn_target()
            self.target_spawn_timer = self.np_random.integers(45, 75)

        # --- Check Termination Conditions ---
        if self.targets_destroyed >= self.WIN_TARGET_COUNT and not self.win:
            self.win = True
            self.game_over = True
            reward += 50
            # sfx: game_win.wav
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_projectiles(self):
        reward = 0
        projectiles_to_add = []
        projectiles_to_remove = []

        for proj in self.projectiles:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]
            proj['lifespan'] -= 1

            # Check for screen boundary split
            if not (0 < proj['pos'][0] < self.WIDTH and proj['pos'][1] > 0) or proj['lifespan'] <= 0:
                if proj['size'] > 2: # Only split if not too small
                    projectiles_to_add.extend(self._split_projectile(proj))
                projectiles_to_remove.append(proj)
                continue

            # Check for collision with targets
            collided_target = None
            for target in self.targets:
                if target['rect'].collidepoint(proj['pos']):
                    collided_target = target
                    break
            
            if collided_target:
                reward += 1.1 # +0.1 for hit, +1.0 for destruction
                self.targets_destroyed += 1
                
                self._create_explosion(collided_target['rect'].center, 30, self.COLOR_TARGET)
                self.targets.remove(collided_target)
                # sfx: explosion.wav
                
                if proj['size'] > 2:
                    projectiles_to_add.extend(self._split_projectile(proj, at_target_pos=collided_target['rect'].center))

                if proj not in projectiles_to_remove:
                    projectiles_to_remove.append(proj)
                
                # Difficulty scaling
                if self.targets_destroyed > 0 and self.targets_destroyed % 5 == 0:
                    self.target_current_speed += 0.2
                    # sfx: level_up.wav
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        self.projectiles.extend(projectiles_to_add)

        return reward

    def _update_targets(self):
        for target in self.targets:
            target['rect'].y += self.target_current_speed
            if target['rect'].bottom > self.HEIGHT - 30:
                if not self.loss: # Ensure penalty is applied only once
                    self.loss = True
                    self.game_over = True
                    # sfx: game_lose.wav
                    return -50
        return 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

    def _launch_projectile(self):
        angle_rad = math.radians(self.staff_angle)
        speed = 8.0
        start_pos = list(self.staff_pos)
        vel = [math.cos(angle_rad) * speed, math.sin(angle_rad) * speed]
        
        self.projectiles.append({
            'pos': start_pos,
            'vel': vel,
            'size': 10,
            'lifespan': 100, # Lifespan in frames
        })

    def _split_projectile(self, parent, at_target_pos=None):
        new_projectiles = []
        split_pos = at_target_pos if at_target_pos else parent['pos']
        
        base_angle = math.degrees(math.atan2(parent['vel'][1], parent['vel'][0]))
        split_angle = 20 # degrees
        
        for i in [-1, 1]:
            angle = math.radians(base_angle + split_angle * i)
            speed = math.hypot(parent['vel'][0], parent['vel'][1]) * 0.9
            
            new_proj = {
                'pos': list(split_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': max(2, parent['size'] * 0.7),
                'lifespan': parent['lifespan'] * 0.8
            }
            new_projectiles.append(new_proj)
        
        self._create_explosion(split_pos, 15, self.COLOR_PROJECTILE)
        # sfx: split.wav
        return new_projectiles

    def _spawn_target(self):
        size = 30
        x = self.np_random.integers(20, self.WIDTH - 20 - size)
        y = -size
        
        # Prevent overlapping spawns
        can_spawn = True
        new_rect = pygame.Rect(x, y, size, size)
        for target in self.targets:
            if new_rect.colliderect(target['rect'].inflate(size, size)):
                can_spawn = False
                break
        
        if can_spawn:
            self.targets.append({'rect': new_rect})

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 25),
                'color': color
            })

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_destroyed": self.targets_destroyed,
            "win": self.win,
            "loss": self.loss,
        }

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render destination line
        pygame.draw.line(self.screen, self.COLOR_DESTINATION_LINE, (0, self.HEIGHT - 30), (self.WIDTH, self.HEIGHT - 30), 3)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 25.0))))
            color = (*p['color'], alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Render targets
        for target in self.targets:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, target['rect'])

        # Render projectiles
        for proj in self.projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            size = int(proj['size'])
            
            # Glow effect
            glow_size = int(size * 1.5)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PROJECTILE, 50), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (x - glow_size, y - glow_size))

            # Core projectile
            pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_PROJECTILE)

        # Render staff
        angle_rad = math.radians(self.staff_angle)
        staff_length = 40
        end_x = self.staff_pos[0] + math.cos(angle_rad) * staff_length
        end_y = self.staff_pos[1] + math.sin(angle_rad) * staff_length
        pygame.draw.line(self.screen, self.COLOR_STAFF, self.staff_pos, (end_x, end_y), 5)
        pygame.draw.circle(self.screen, self.COLOR_STAFF, self.staff_pos, 8)
        
        # Render cooldown indicator
        cooldown_ratio = self.launch_cooldown / self.launch_cooldown_max
        if cooldown_ratio > 0:
            arc_rect = pygame.Rect(self.staff_pos[0]-12, self.staff_pos[1]-12, 24, 24)
            start_angle = math.pi / 2
            end_angle = start_angle - (cooldown_ratio * 2 * math.pi)
            pygame.draw.arc(self.screen, self.COLOR_PROJECTILE, arc_rect, start_angle, end_angle, 2)


        # Render UI
        self._render_ui()

    def _render_ui(self):
        # Targets destroyed
        target_text = f"DESTROYED: {self.targets_destroyed} / {self.WIN_TARGET_COUNT}"
        text_surf = self.font_small.render(target_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Steps
        steps_text = f"STEPS: {self.steps} / {self.MAX_STEPS}"
        text_surf = self.font_small.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER_TEXT
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

# Example usage for interactive play
if __name__ == "__main__":
    # Un-comment the line below to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    
    # This part of the script is for human play and requires a display.
    # It will not run in a headless environment unless you change the SDL_VIDEODRIVER.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Recursive Annihilator")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        while not terminated:
            # --- Human Input ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Rendering ---
            # The observation is already the rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Clock ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            clock.tick(env.FPS)

            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Targets Destroyed: {info['targets_destroyed']}")
                pygame.time.wait(3000) # Pause for 3 seconds before closing

    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This script is designed for a headless environment. To run with a display,")
        print("you may need to install additional dependencies (e.g., 'pip install pygame')")
        print("and comment out the 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' line.")

    finally:
        env.close()