import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set for headless operation, required by the verifier
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to move silently. Hold Space to sprint."
    )

    game_description = (
        "Evade a relentless killer in a procedurally generated house. "
        "Find the green exit before the time runs out or you are caught. "
        "Moving creates noise that attracts the killer."
    )

    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 50, 60)
    COLOR_ROOM_FLOOR = (25, 35, 45)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_KILLER = (200, 30, 30)
    COLOR_EXIT = (50, 220, 50)
    COLOR_LIGHT = (255, 230, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_NOISE = (255, 255, 255)
    
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 180
    MAX_STEPS = 1000 * 30 # A very high number, timeout is the primary limit

    PLAYER_SIZE = 10
    PLAYER_SPEED = 2.5
    KILLER_SIZE = 12
    KILLER_SPEED_PATROL = 1.5
    KILLER_SPEED_HUNT = 2.8
    KILLER_CATCH_DISTANCE = 15
    KILLER_HEARING_RANGE = 200

    class Particle:
        def __init__(self, pos, start_radius, max_radius, lifetime):
            self.pos = pos
            self.radius = start_radius
            self.max_radius = max_radius
            self.lifetime = lifetime
            self.life = lifetime

        def update(self):
            self.life -= 1
            progress = (self.lifetime - self.life) / self.lifetime
            self.radius = self.max_radius * progress
            return self.life > 0

        def draw(self, surface, offset):
            if self.life <= 0:
                return
            alpha = int(200 * (self.life / self.lifetime))
            if alpha <= 0:
                return
            
            pos_on_screen = (int(self.pos[0] + offset[0]), int(self.pos[1] + offset[1]))
            
            # Use gfxdraw for anti-aliased circles
            pygame.gfxdraw.aacircle(surface, pos_on_screen[0], pos_on_screen[1], int(self.radius), (*GameEnv.COLOR_NOISE, alpha))


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
        self.font_small = pygame.font.SysFont("sans-serif", 20)
        self.font_large = pygame.font.SysFont("sans-serif", 50)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        # State variables to be initialized in reset()
        self.player_pos = pygame.Vector2(0, 0)
        self.killer_pos = pygame.Vector2(0, 0)
        self.exit_pos = pygame.Rect(0, 0, 0, 0)
        self.walls = []
        self.rooms = []
        self.time_left = 0
        self.noise_level = 0
        self.particles = []
        self.lights = []
        self.killer_state = "PATROL"
        self.killer_target = pygame.Vector2(0, 0)
        self.killer_patrol_path = []
        self.killer_patrol_index = 0
        self.killer_was_hunting = False
        self.player_current_room_idx = -1

        # self.validate_implementation() # Removed from __init__ to avoid issues with verifier
        self.reset()

    def _generate_layout(self):
        self.walls = []
        self.rooms = []
        self.lights = []
        
        # Room generation
        min_rooms, max_rooms = 8, 12
        num_rooms = self.np_random.integers(min_rooms, max_rooms + 1)
        
        for _ in range(num_rooms * 5): # Try 5x to get enough rooms
            if len(self.rooms) >= num_rooms:
                break
            
            w = self.np_random.integers(60, 150)
            h = self.np_random.integers(60, 150)
            x = self.np_random.integers(-800, 800)
            y = self.np_random.integers(-600, 600)
            new_room = pygame.Rect(x, y, w, h)
            
            if not any(new_room.colliderect(r.inflate(40, 40)) for r in self.rooms):
                self.rooms.append(new_room)
                if self.np_random.random() < 0.4: # 40% chance of a light
                    light_pos = new_room.center
                    light_radius = self.np_random.integers(80, 120)
                    self.lights.append((light_pos, light_radius))
        
        if not self.rooms: # Failsafe
            self.rooms.append(pygame.Rect(-100, -100, 200, 200))

        # Connect rooms with corridors
        connected = {0}
        to_connect = list(range(1, len(self.rooms)))
        while to_connect:
            i = self.np_random.choice(list(connected))
            
            # Find closest unconnected room
            closest_dist = float('inf')
            closest_j = -1
            for j in to_connect:
                dist = pygame.Vector2(self.rooms[i].center).distance_to(self.rooms[j].center)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_j = j
            
            if closest_j != -1:
                p1 = pygame.Vector2(self.rooms[i].center)
                p2 = pygame.Vector2(self.rooms[closest_j].center)
                
                corridor_width = 20
                if self.np_random.random() > 0.5: # H-then-V
                    h_corr = pygame.Rect(min(p1.x, p2.x) - corridor_width/2, p1.y - corridor_width/2, abs(p1.x - p2.x) + corridor_width, corridor_width)
                    v_corr = pygame.Rect(p2.x - corridor_width/2, min(p1.y, p2.y) - corridor_width/2, corridor_width, abs(p1.y - p2.y) + corridor_width)
                else: # V-then-H
                    v_corr = pygame.Rect(p1.x - corridor_width/2, min(p1.y, p2.y) - corridor_width/2, corridor_width, abs(p1.y - p2.y) + corridor_width)
                    h_corr = pygame.Rect(min(p1.x, p2.x) - corridor_width/2, p2.y - corridor_width/2, abs(p1.x - p2.x) + corridor_width, corridor_width)
                
                self.rooms.extend([h_corr, v_corr])
                connected.add(closest_j)
                to_connect.remove(closest_j)

        # Create walls from room rects
        wall_thickness = 4
        all_floors = self.rooms
        self.rooms = [r for r in self.rooms if r.width > 25 and r.height > 25] # Filter out corridors for logic
        
        for r in all_floors:
            self.walls.append(pygame.Rect(r.left, r.top, r.width, wall_thickness))
            self.walls.append(pygame.Rect(r.left, r.bottom - wall_thickness, r.width, wall_thickness))
            self.walls.append(pygame.Rect(r.left, r.top, wall_thickness, r.height))
            self.walls.append(pygame.Rect(r.right - wall_thickness, r.top, wall_thickness, r.height))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_layout()
        
        # Place player, exit, killer
        start_room_idx = self.np_random.choice(len(self.rooms))
        self.player_pos = pygame.Vector2(self.rooms[start_room_idx].center)
        self.player_current_room_idx = start_room_idx

        # Find a distant room for the exit
        distances = [self.player_pos.distance_to(r.center) for r in self.rooms]
        exit_room_idx = np.argmax(distances)
        self.exit_pos = pygame.Rect(0,0,30,30)
        self.exit_pos.center = self.rooms[exit_room_idx].center

        # Killer setup
        possible_killer_starts = [i for i, r in enumerate(self.rooms) if i != start_room_idx and self.player_pos.distance_to(r.center) > 200]
        if not possible_killer_starts: possible_killer_starts = [i for i in range(len(self.rooms)) if i != start_room_idx]
        killer_start_idx = self.np_random.choice(possible_killer_starts)
        self.killer_pos = pygame.Vector2(self.rooms[killer_start_idx].center)
        
        self._update_killer_patrol_path(start_room_idx)
        self.killer_state = "PATROL"
        self.killer_target = pygame.Vector2(self.killer_patrol_path[0])
        
        # Reset state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_left = self.GAME_DURATION_SECONDS * self.FPS
        self.noise_level = 0
        self.particles = []
        self.killer_was_hunting = False
        
        return self._get_observation(), self._get_info()

    def _update_killer_patrol_path(self, player_room_idx):
        patrol_rooms = [i for i, r in enumerate(self.rooms) if i != player_room_idx and i != self.player_current_room_idx]
        self.np_random.shuffle(patrol_rooms)
        self.killer_patrol_path = [pygame.Vector2(self.rooms[i].center) for i in patrol_rooms[:4]]
        if not self.killer_patrol_path: # Failsafe
            self.killer_patrol_path = [self.killer_pos]
        self.killer_patrol_index = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty for time passing
        
        # --- Player Movement ---
        player_speed = self.PLAYER_SPEED
        noise_multiplier = 1.0
        moved = False

        if shift_held:
            player_speed *= 0.6
            noise_multiplier *= 0.1
        elif space_held:
            player_speed *= 1.8
            noise_multiplier *= 2.5

        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.scale_to_length(player_speed)
            moved = True
            
            # Collision handling
            new_pos_x = self.player_pos + pygame.Vector2(move_vec.x, 0)
            player_rect_x = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect_x.center = new_pos_x
            if player_rect_x.collidelist(self.walls) == -1:
                self.player_pos = new_pos_x

            new_pos_y = self.player_pos + pygame.Vector2(0, move_vec.y)
            player_rect_y = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect_y.center = new_pos_y
            if player_rect_y.collidelist(self.walls) == -1:
                self.player_pos = new_pos_y

        # --- Noise Generation ---
        self.noise_level = max(0, self.noise_level - 0.5)
        if moved:
            # sfx: player_footstep()
            noise_generated = 5 * noise_multiplier
            self.noise_level += noise_generated
            if noise_generated > 1:
                # FIX: pygame.Vector2 doesn't have .copy(). Use constructor to copy.
                self.particles.append(self.Particle(pygame.Vector2(self.player_pos), 5, self.noise_level * 3, 30))

        # --- Room Detection & Reward ---
        new_room_found = False
        for i, room in enumerate(self.rooms):
            if room.collidepoint(self.player_pos):
                if i != self.player_current_room_idx:
                    self.player_current_room_idx = i
                    new_room_found = True
                break
        if new_room_found:
            reward += 0.2

        # --- Killer AI ---
        dist_to_player = self.killer_pos.distance_to(self.player_pos)
        
        if self.noise_level > 10 and dist_to_player < self.KILLER_HEARING_RANGE:
            if self.killer_state != "HUNT":
                # sfx: killer_alerted()
                self.killer_was_hunting = True
            self.killer_state = "HUNT"
            # FIX: pygame.Vector2 doesn't have .copy(). Use constructor to copy.
            self.killer_target = pygame.Vector2(self.player_pos)

        if self.killer_state == "HUNT":
            if self.killer_pos.distance_to(self.killer_target) < 10:
                self.killer_state = "PATROL"
                self._update_killer_patrol_path(self.player_current_room_idx)
                self.killer_target = self.killer_patrol_path[0]
                if self.killer_was_hunting:
                    reward += 5.0 # Successful hide reward
                    self.killer_was_hunting = False
            else:
                move_dir = (self.killer_target - self.killer_pos).normalize()
                self.killer_pos += move_dir * self.KILLER_SPEED_HUNT
        
        elif self.killer_state == "PATROL":
            if self.killer_pos.distance_to(self.killer_target) < 10:
                self.killer_patrol_index = (self.killer_patrol_index + 1) % len(self.killer_patrol_path)
                self.killer_target = self.killer_patrol_path[self.killer_patrol_index]
            move_dir = (self.killer_target - self.killer_pos).normalize()
            self.killer_pos += move_dir * self.KILLER_SPEED_PATROL

        # --- Game State Update ---
        self.steps += 1
        self.time_left -= 1
        self.particles = [p for p in self.particles if p.update()]
        
        # --- Termination Check ---
        terminated = False
        player_rect = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos

        if dist_to_player < self.KILLER_CATCH_DISTANCE:
            # sfx: player_caught_scream()
            reward = -100
            terminated = True
            self.game_over = True
            self.win_message = "CAUGHT"
        elif player_rect.colliderect(self.exit_pos):
            # sfx: win_sound()
            reward = 100
            terminated = True
            self.game_over = True
            self.win_message = "ESCAPED!"
        elif self.time_left <= 0:
            reward = -50
            terminated = True
            self.game_over = True
            self.win_message = "TIME'S UP"
        elif self.steps >= self.MAX_STEPS:
            reward = -50
            terminated = True
            self.game_over = True
            self.win_message = "STEP LIMIT"

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        camera_offset = pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2) - self.player_pos

        # Draw floors
        for r in self.rooms:
            pygame.draw.rect(self.screen, self.COLOR_ROOM_FLOOR, r.move(camera_offset))
        
        # Draw lights
        for pos, radius in self.lights:
            flicker = 1.0 + math.sin(self.steps * 0.2 + pos[0]) * 0.1
            r = int(radius * flicker)
            
            light_surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(light_surf, (*self.COLOR_LIGHT, 30), (r, r), r)
            self.screen.blit(light_surf, (pos[0] - r + camera_offset.x, pos[1] - r + camera_offset.y), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall.move(camera_offset))
        
        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_pos.move(camera_offset))
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen, camera_offset)

        # Draw killer
        killer_screen_pos = self.killer_pos + camera_offset
        killer_aura_radius = self.KILLER_SIZE * (2.5 + math.sin(self.steps * 0.1) * 0.5)
        aura_surf = pygame.Surface((killer_aura_radius*2, killer_aura_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(aura_surf, (*self.COLOR_KILLER, 40), (killer_aura_radius, killer_aura_radius), int(killer_aura_radius))
        self.screen.blit(aura_surf, (killer_screen_pos.x - killer_aura_radius, killer_screen_pos.y - killer_aura_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(self.screen, self.COLOR_KILLER, (int(killer_screen_pos.x), int(killer_screen_pos.y)), self.KILLER_SIZE)

        # Draw player (always at center)
        player_center_screen = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_center_screen, self.PLAYER_SIZE)
        
        # Draw noise indicator around player
        if self.noise_level > 1:
            noise_indicator_radius = self.PLAYER_SIZE + 5
            noise_angle = (self.noise_level / 100) * 360
            if noise_angle > 0:
                arc_rect = pygame.Rect(0,0, noise_indicator_radius*2, noise_indicator_radius*2)
                arc_rect.center = player_center_screen
                pygame.draw.arc(self.screen, self.COLOR_NOISE, arc_rect, 0, math.radians(min(359.9, noise_angle)), 2)

        # Draw UI
        time_text = f"TIME: {max(0, self.time_left // self.FPS):03d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        if self.game_over:
            end_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left // self.FPS,
            "killer_state": self.killer_state,
            "noise_level": self.noise_level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    # This will override the "dummy" setting for the video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy' or 'windows'
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Escape The House")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose observation for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(GameEnv.FPS)
        
    env.close()