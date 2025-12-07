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
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Launch your core through abstract levels, collecting fragments to unlock the path to the exit while avoiding deadly nightmares."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to aim. Press space to launch. Collect fragments to unlock paths and reach the exit."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_WALL = (40, 30, 70)
        self.COLOR_HAZARD_WALL = (100, 20, 40)
        self.COLOR_UNLOCKABLE_WALL = (60, 50, 100)
        self.COLOR_UNLOCKABLE_WALL_READY = (80, 70, 130)
        
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        
        self.COLOR_FRAGMENT = (255, 255, 0)
        self.COLOR_FRAGMENT_GLOW = (150, 150, 0)
        
        self.COLOR_EXIT = (255, 255, 150)
        self.COLOR_EXIT_GLOW = (150, 150, 50)
        
        self.COLOR_NIGHTMARE = (200, 0, 0)
        self.COLOR_NIGHTMARE_GLOW = (100, 0, 0)
        
        self.COLOR_AIM = (255, 255, 255, 150)
        self.COLOR_TEXT = (220, 220, 240)

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Initialize state variables
        self.level_idx = -1 # Start at -1 so first level is 0
        self.player = None
        self.fragments = []
        self.walls = []
        self.hazard_walls = []
        self.unlockable_walls = []
        self.nightmares = []
        self.exit = None
        self.particles = []
        self.game_phase = "aiming" # 'aiming' or 'moving'
        self.aim_angle = 0
        self.steps = 0
        self.score = 0
        self.fragments_collected = 0
        self.prev_space_held = False
        self.reward_this_step = 0

    def _generate_level(self):
        self.walls = []
        self.hazard_walls = []
        self.fragments = []
        self.unlockable_walls = []
        self.nightmares = []

        # Level progression
        self.level_idx = (self.level_idx + 1) % 3
        nightmare_speed = 1.0 + self.level_idx * 0.5

        if self.level_idx == 1: # Level 1
            self.player = {'pos': pygame.math.Vector2(100, 200), 'radius': 12}
            self.exit = {'pos': pygame.math.Vector2(540, 200), 'radius': 15}
            self.walls.extend([pygame.Rect(0,0,self.WIDTH,10), pygame.Rect(0,self.HEIGHT-10,self.WIDTH,10), pygame.Rect(0,0,10,self.HEIGHT), pygame.Rect(self.WIDTH-10,0,10,self.HEIGHT)])
            self.fragments.extend([{'pos': pygame.math.Vector2(200, 100), 'collected': False}, {'pos': pygame.math.Vector2(320, 300), 'collected': False}])
            self.nightmares.append({'path': [pygame.math.Vector2(400, 50), pygame.math.Vector2(400, 350)], 'pos': pygame.math.Vector2(400, 50), 'radius': 15, 'speed': nightmare_speed, 'direction': 1})
        elif self.level_idx == 2: # Level 2
            self.player = {'pos': pygame.math.Vector2(60, 60), 'radius': 12}
            self.exit = {'pos': pygame.math.Vector2(580, 340), 'radius': 15}
            self.walls.extend([pygame.Rect(0,0,self.WIDTH,10), pygame.Rect(0,self.HEIGHT-10,self.WIDTH,10), pygame.Rect(0,0,10,self.HEIGHT), pygame.Rect(self.WIDTH-10,0,10,self.HEIGHT)])
            self.walls.extend([pygame.Rect(150, 0, 20, 250), pygame.Rect(450, 150, 20, 250)])
            self.hazard_walls.append(pygame.Rect(250, 180, 150, 20))
            self.fragments.extend([{'pos': pygame.math.Vector2(320, 100), 'collected': False}, {'pos': pygame.math.Vector2(80, 320), 'collected': False}])
            self.unlockable_walls.append({'rect': pygame.Rect(500, 0, 20, 300), 'fragments_needed': 2, 'unlocked': False})
            self.nightmares.append({'path': [pygame.math.Vector2(220, 50), pygame.math.Vector2(220, 350)], 'pos': pygame.math.Vector2(220, 350), 'radius': 15, 'speed': nightmare_speed, 'direction': -1})
            self.nightmares.append({'path': [pygame.math.Vector2(380, 50), pygame.math.Vector2(380, 350)], 'pos': pygame.math.Vector2(380, 50), 'radius': 15, 'speed': nightmare_speed, 'direction': 1})
        else: # Level 0 (Default)
            self.player = {'pos': pygame.math.Vector2(60, self.HEIGHT / 2), 'radius': 12}
            self.exit = {'pos': pygame.math.Vector2(self.WIDTH - 60, self.HEIGHT / 2), 'radius': 15}
            self.walls.extend([pygame.Rect(0,0,self.WIDTH,10), pygame.Rect(0,self.HEIGHT-10,self.WIDTH,10), pygame.Rect(0,0,10,self.HEIGHT), pygame.Rect(self.WIDTH-10,0,10,self.HEIGHT)])
            self.walls.append(pygame.Rect(self.WIDTH/2 - 10, 100, 20, self.HEIGHT - 200))
            self.fragments.append({'pos': pygame.math.Vector2(self.WIDTH/2, 60), 'collected': False})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        self.steps = 0
        self.score = 0
        self.fragments_collected = 0
        self.game_phase = "aiming"
        self.aim_angle = 0
        self.prev_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.reward_this_step -= 0.001 # Cost of living
        
        # Update nightmares
        for n in self.nightmares:
            p1, p2 = n['path']
            target = p2 if n['direction'] == 1 else p1
            move_vec = (target - n['pos'])
            if move_vec.length_squared() < (n['speed'] ** 2):
                n['pos'] = target
                n['direction'] *= -1
            else:
                n['pos'] += move_vec.normalize() * n['speed']

        if self.game_phase == 'aiming':
            # Adjust aim
            if movement == 1: self.aim_angle -= 0.05 # Up
            elif movement == 2: self.aim_angle += 0.05 # Down
            elif movement == 3: self.aim_angle -= 0.05 # Left (same as up for circular)
            elif movement == 4: self.aim_angle += 0.05 # Right (same as down for circular)
            self.aim_angle %= (2 * math.pi)

            # Launch
            if space_held and not self.prev_space_held:
                self.game_phase = 'moving'
                self.player['vel'] = pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 8
                self._create_particles(self.player['pos'], 20, self.COLOR_PLAYER, 1, 3) # Launch burst
                # sfx: launch_sound

        elif self.game_phase == 'moving':
            # Move player
            new_pos = self.player['pos'] + self.player['vel']
            
            # --- Collision Detection ---
            collided = False
            
            # Walls
            for wall in self.walls:
                if self._check_circle_rect_collision(new_pos, self.player['radius'], wall):
                    collided = True; break
            if not collided:
                for wall_data in self.unlockable_walls:
                    if not wall_data['unlocked'] and self._check_circle_rect_collision(new_pos, self.player['radius'], wall_data['rect']):
                        collided = True; break
            
            if collided:
                self.game_phase = 'aiming'
                self.player['pos'] = self._find_safe_spot(self.player['pos'], self.player['vel'])
                self._create_particles(self.player['pos'], 10, self.COLOR_WALL, 0.5, 2) # Wall hit
                # sfx: wall_thud
            else:
                self.player['pos'] = new_pos

            # Hazard Walls
            for wall in self.hazard_walls:
                if self._check_circle_rect_collision(self.player['pos'], self.player['radius'], wall):
                    self.reward_this_step -= 100
                    terminated = True
                    self._create_particles(self.player['pos'], 50, self.COLOR_HAZARD_WALL, 1, 5, 20) # Death burst
                    # sfx: player_death_sfx
                    break
            if terminated:
                pass # Skip other checks
            
            # Fragments
            else:
                for frag in self.fragments:
                    if not frag['collected'] and (self.player['pos'] - frag['pos']).length_squared() < (self.player['radius'] + 5)**2:
                        frag['collected'] = True
                        self.fragments_collected += 1
                        self.reward_this_step += 1.0 # Changed from 0.1 for more impact
                        self._create_particles(frag['pos'], 30, self.COLOR_FRAGMENT, 1, 4) # Collect burst
                        # sfx: collect_fragment_sfx

                        # Check for unlocking walls
                        for wall_data in self.unlockable_walls:
                            if not wall_data['unlocked'] and self.fragments_collected >= wall_data['fragments_needed']:
                                wall_data['unlocked'] = True
                                self.reward_this_step += 5.0
                                self._create_particles(pygame.math.Vector2(wall_data['rect'].center), 50, self.COLOR_UNLOCKABLE_WALL_READY, 2, 5) # Unlock effect
                                # sfx: unlock_path_sfx

            # Nightmares
            if not terminated:
                for n in self.nightmares:
                    if (self.player['pos'] - n['pos']).length_squared() < (self.player['radius'] + n['radius'])**2:
                        self.reward_this_step -= 100
                        terminated = True
                        self._create_particles(self.player['pos'], 50, self.COLOR_NIGHTMARE, 1, 5, 20) # Death burst
                        # sfx: nightmare_catch_sfx
                        break
            
            # Exit
            if not terminated:
                if (self.player['pos'] - self.exit['pos']).length_squared() < (self.player['radius'] + self.exit['radius'])**2:
                    self.reward_this_step += 100
                    terminated = True
                    # sfx: level_complete_sfx
            
            # World bounds
            if not (0 < self.player['pos'].x < self.WIDTH and 0 < self.player['pos'].y < self.HEIGHT):
                self.game_phase = 'aiming'
                self.player['pos'] = self._find_safe_spot(self.player['pos'], self.player['vel'])

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        self.prev_space_held = space_held
        self.score += self.reward_this_step
        
        # Check termination conditions
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _find_safe_spot(self, pos, vel):
        # Step back from collision
        safe_pos = pygame.math.Vector2(pos)
        for _ in range(20):
            safe_pos -= vel.normalize()
            collided = False
            for wall in self.walls:
                if self._check_circle_rect_collision(safe_pos, self.player['radius'], wall):
                    collided = True; break
            if not collided:
                for wall_data in self.unlockable_walls:
                    if not wall_data['unlocked'] and self._check_circle_rect_collision(safe_pos, self.player['radius'], wall_data['rect']):
                        collided = True; break
            if not collided:
                return safe_pos
        return safe_pos # Failsafe

    def _check_circle_rect_collision(self, circle_pos, circle_radius, rect):
        closest_x = max(rect.left, min(circle_pos.x, rect.right))
        closest_y = max(rect.top, min(circle_pos.y, rect.bottom))
        distance_x = circle_pos.x - closest_x
        distance_y = circle_pos.y - closest_y
        return (distance_x**2 + distance_y**2) < (circle_radius**2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        for wall in self.hazard_walls:
            pygame.draw.rect(self.screen, self.COLOR_HAZARD_WALL, wall)
        for wall_data in self.unlockable_walls:
            if not wall_data['unlocked']:
                color = self.COLOR_UNLOCKABLE_WALL_READY if self.fragments_collected >= wall_data['fragments_needed'] else self.COLOR_UNLOCKABLE_WALL
                pygame.draw.rect(self.screen, color, wall_data['rect'])
                # Add a subtle flicker/pulse effect
                if color == self.COLOR_UNLOCKABLE_WALL_READY:
                    alpha = 10 + math.sin(self.steps * 0.2) * 10
                    s = pygame.Surface(wall_data['rect'].size, pygame.SRCALPHA)
                    s.fill((255, 255, 255, alpha))
                    self.screen.blit(s, wall_data['rect'].topleft)

        # Nightmares
        for n in self.nightmares:
            self._draw_glowing_circle(n['pos'], n['radius'], self.COLOR_NIGHTMARE, self.COLOR_NIGHTMARE_GLOW)
            # Add a distortion effect
            for i in range(5):
                offset_angle = self.steps * 0.1 + i * (2 * math.pi / 5)
                offset_mag = 5 + math.sin(self.steps * 0.3 + i) * 3
                p1 = n['pos'] + pygame.math.Vector2(math.cos(offset_angle), math.sin(offset_angle)) * (n['radius'] + offset_mag)
                p2 = n['pos'] + pygame.math.Vector2(math.cos(offset_angle + 0.5), math.sin(offset_angle + 0.5)) * (n['radius'] + offset_mag - 2)
                pygame.draw.line(self.screen, self.COLOR_NIGHTMARE, p1, p2, 2)

        # Fragments
        for frag in self.fragments:
            if not frag['collected']:
                pulse = 1 + math.sin(self.steps * 0.1 + frag['pos'].x) * 0.5
                self._draw_glowing_circle(frag['pos'], 5 * pulse, self.COLOR_FRAGMENT, self.COLOR_FRAGMENT_GLOW)
        
        # Exit
        self._draw_glowing_circle(self.exit['pos'], self.exit['radius'], self.COLOR_EXIT, self.COLOR_EXIT_GLOW)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, (p['life'] / p['max_life']) * 255))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Player
        self._draw_glowing_circle(self.player['pos'], self.player['radius'], self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Aiming reticle
        if self.game_phase == 'aiming':
            start_pos = self.player['pos']
            for i in range(15):
                end_pos = start_pos + pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 15
                alpha = 150 - i * 8
                pygame.draw.line(self.screen, (*self.COLOR_AIM[:3], alpha), start_pos, end_pos, 2)
                start_pos = end_pos + pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 5

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        # Draw multiple layers for a bloom effect
        for i in range(4, 0, -1):
            alpha = 30 - i * 5
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius + i * 3), (*glow_color, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius + i * 3), (*glow_color, alpha))
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifetime=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': lifetime,
                'max_life': lifetime,
                'color': color,
                'radius': self.np_random.integers(2, 5)
            })

    def _render_ui(self):
        total_frags = len(self.fragments)
        frags_text = f"Fragments: {self.fragments_collected} / {total_frags}"
        score_text = f"Score: {self.score:.1f}"
        
        frags_surface = self.font.render(frags_text, True, self.COLOR_TEXT)
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(frags_surface, (15, 15))
        self.screen.blit(score_surface, (15, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fragments_collected": self.fragments_collected,
            "level": self.level_idx
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It is not used by the evaluation environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Create a display surface
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv")
    
    running = True
    terminated = False
    
    # --- Human player controls ---
    # Use arrow keys to aim, space to launch
    aim_direction = 0 # 0: none, 1: up/left, 2: down/right
    launch = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_LEFT:
                    aim_direction = 1
                elif event.key == pygame.K_DOWN or event.key == pygame.K_RIGHT:
                    aim_direction = 2
                if event.key == pygame.K_SPACE:
                    launch = True
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset(seed=42)
                    print("Environment reset.")
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT]:
                    aim_direction = 0

        # Map human input to action space
        movement_action = 0
        if aim_direction == 1:
            movement_action = 1 # Corresponds to up
        elif aim_direction == 2:
            movement_action = 2 # Corresponds to down

        space_action = 1 if launch else 0
        shift_action = 0
        
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the screen for human viewing
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        # Reset for next launch
        launch = False

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(seed=42)
            pygame.time.wait(1000)

        env.clock.tick(env.FPS)
        
    env.close()